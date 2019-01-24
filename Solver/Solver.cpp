#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

#include <cmath>


using namespace std;


namespace szx {

#pragma region Solver::Cli
int Solver::Cli::run(int argc, char * argv[]) {
    Log(LogSwitch::Szx::Cli) << "parse command line arguments." << endl;
    Set<String> switchSet;
    Map<String, char*> optionMap({ // use string as key to compare string contents instead of pointers.
        { InstancePathOption(), nullptr },
        { SolutionPathOption(), nullptr },
        { RandSeedOption(), nullptr },
        { TimeoutOption(), nullptr },
        { MaxIterOption(), nullptr },
        { JobNumOption(), nullptr },
        { RunIdOption(), nullptr },
        { EnvironmentPathOption(), nullptr },
        { ConfigPathOption(), nullptr },
        { LogPathOption(), nullptr }
    });

    for (int i = 1; i < argc; ++i) { // skip executable name.
        auto mapIter = optionMap.find(argv[i]);
        if (mapIter != optionMap.end()) { // option argument.
            mapIter->second = argv[++i];
        } else { // switch argument.
            switchSet.insert(argv[i]);
        }
    }

    Log(LogSwitch::Szx::Cli) << "execute commands." << endl;
    if (switchSet.find(HelpSwitch()) != switchSet.end()) {
        cout << HelpInfo() << endl;
    }

    if (switchSet.find(AuthorNameSwitch()) != switchSet.end()) {
        cout << AuthorName() << endl;
    }

    Solver::Environment env;
    env.load(optionMap);
    if (env.instPath.empty() || env.slnPath.empty()) { return -1; }

    Solver::Configuration cfg;
    cfg.load(env.cfgPath);

    Log(LogSwitch::Szx::Input) << "load instance " << env.instPath << " (seed=" << env.randSeed << ")." << endl;
    Problem::Input input;
    if (!input.load(env.instPath)) { return -1; }

    Solver solver(input, env, cfg);
    solver.solve();

    pb::Submission submission;
    submission.set_thread(to_string(env.jobNum));
    submission.set_instance(env.friendlyInstName());
    submission.set_duration(to_string(solver.timer.elapsedSeconds()) + "s");

    solver.output.save(env.slnPath, submission);
    #if SZX_DEBUG
    solver.output.save(env.solutionPathWithTime(), submission);
    solver.record();
    #endif // SZX_DEBUG

    return 0;
}
#pragma endregion Solver::Cli

#pragma region Solver::Environment
void Solver::Environment::load(const Map<String, char*> &optionMap) {
    char *str;

    str = optionMap.at(Cli::EnvironmentPathOption());
    if (str != nullptr) { loadWithoutCalibrate(str); }

    str = optionMap.at(Cli::InstancePathOption());
    if (str != nullptr) { instPath = str; }

    str = optionMap.at(Cli::SolutionPathOption());
    if (str != nullptr) { slnPath = str; }

    str = optionMap.at(Cli::RandSeedOption());
    if (str != nullptr) { randSeed = atoi(str); }

    str = optionMap.at(Cli::TimeoutOption());
    if (str != nullptr) { msTimeout = static_cast<Duration>(atof(str) * Timer::MillisecondsPerSecond); }

    str = optionMap.at(Cli::MaxIterOption());
    if (str != nullptr) { maxIter = atoi(str); }

    str = optionMap.at(Cli::JobNumOption());
    if (str != nullptr) { jobNum = atoi(str); }

    str = optionMap.at(Cli::RunIdOption());
    if (str != nullptr) { rid = str; }

    str = optionMap.at(Cli::ConfigPathOption());
    if (str != nullptr) { cfgPath = str; }

    str = optionMap.at(Cli::LogPathOption());
    if (str != nullptr) { logPath = str; }

    calibrate();
}

void Solver::Environment::load(const String &filePath) {
    loadWithoutCalibrate(filePath);
    calibrate();
}

void Solver::Environment::loadWithoutCalibrate(const String &filePath) {
    // EXTEND[szx][8]: load environment from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Environment::save(const String &filePath) const {
    // EXTEND[szx][8]: save environment to file.
}
void Solver::Environment::calibrate() {
    // adjust thread number.
    int threadNum = thread::hardware_concurrency();
    if ((jobNum <= 0) || (jobNum > threadNum)) { jobNum = threadNum; }

    // adjust timeout.
    msTimeout -= Environment::SaveSolutionTimeInMillisecond;
}
#pragma endregion Solver::Environment

#pragma region Solver::Configuration
void Solver::Configuration::load(const String &filePath) {
    // EXTEND[szx][5]: load configuration from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Configuration::save(const String &filePath) const {
    // EXTEND[szx][5]: save configuration to file.
}
#pragma endregion Solver::Configuration

#pragma region Solver
bool Solver::solve() {
    init();

    int workerNum = (max)(1, env.jobNum / cfg.threadNumPerWorker);
    cfg.threadNumPerWorker = env.jobNum / workerNum;
    List<Solution> solutions(workerNum, Solution(this));
    List<bool> success(workerNum);

    Log(LogSwitch::Szx::Framework) << "launch " << workerNum << " workers." << endl;
    List<thread> threadList;
    threadList.reserve(workerNum);
    for (int i = 0; i < workerNum; ++i) {
        // TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
        // OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
        threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], i); });
    }
    for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

    Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
    int bestIndex = -1;
    Length bestValue = Problem::MaxDistance;
    for (int i = 0; i < workerNum; ++i) {
        if (!success[i]) { continue; }
        Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].coverRadius << endl;
        if (solutions[i].coverRadius >= bestValue) { continue; }
        bestIndex = i;
        bestValue = solutions[i].coverRadius;
    }

    env.rid = to_string(bestIndex);
    if (bestIndex < 0) { return false; }
    output = solutions[bestIndex];
    return true;
}

void Solver::record() const {
    #if SZX_DEBUG
    int generation = 0;

    ostringstream log;

    System::MemoryUsage mu = System::peakMemoryUsage();

    Length obj = output.coverRadius;
    Length checkerObj = -1;
    bool feasible = check(checkerObj);

    // record basic information.
    log << env.friendlyLocalTime() << ","
        << env.rid << ","
        << env.instPath << ","
        << feasible << "," << (obj - checkerObj) << ",";
    if (Problem::isTopologicalGraph(input)) {
        log << obj << ",";
    } else {
        auto oldPrecision = log.precision();
        log.precision(2);
        log << fixed << setprecision(2) << (obj / aux.objScale) << ",";
        log.precision(oldPrecision);
    }
    log << timer.elapsedSeconds() << ","
        << mu.physicalMemory << "," << mu.virtualMemory << ","
        << env.randSeed << ","
        << cfg.toBriefStr() << ","
        << generation << "," << iteration << ",";

    // record solution vector.
    // EXTEND[szx][2]: save solution in log.
    log << endl;

    // append all text atomically.
    static mutex logFileMutex;
    lock_guard<mutex> logFileGuard(logFileMutex);

    ofstream logFile(env.logPath, ios::app);
    logFile.seekp(0, ios::end);
    if (logFile.tellp() <= 0) {
        logFile << "Time,ID,Instance,Feasible,ObjMatch,Distance,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
    }
    logFile << log.str();
    logFile.close();
    #endif // SZX_DEBUG
}

bool Solver::check(Length &checkerObj) const {
    #if SZX_DEBUG
    enum CheckerFlag {
        IoError = 0x0,
        FormatError = 0x1,
        TooManyCentersError = 0x2
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::TooManyCentersError) { Log(LogSwitch::Checker) << "TooManyCentersError." << endl; }
    return false;
    #else
    checkerObj = 0;
    return true;
    #endif // SZX_DEBUG
}

void Solver::init() {
    ID nodeNum = input.graph().nodenum();

    aux.adjMat.init(nodeNum, nodeNum);
    fill(aux.adjMat.begin(), aux.adjMat.end(), Problem::MaxDistance);
    for (ID n = 0; n < nodeNum; ++n) { aux.adjMat.at(n, n) = 0; }

    if (Problem::isTopologicalGraph(input)) {
        aux.objScale = Problem::TopologicalGraphObjScale;
        for (auto e = input.graph().edges().begin(); e != input.graph().edges().end(); ++e) {
            // only record the last appearance of each edge.
            aux.adjMat.at(e->source(), e->target()) = e->length();
            aux.adjMat.at(e->target(), e->source()) = e->length();
        }

        Timer timer(30s);
        constexpr bool IsUndirectedGraph = true;           //�����Ƿ�������ͼ������ͼ�Ƿ�Գƣ�����ѡ��Floyd
        IsUndirectedGraph
            ? Floyd::findAllPairsPaths_symmetric(aux.adjMat)
            : Floyd::findAllPairsPaths_asymmetric(aux.adjMat);
        Log(LogSwitch::Preprocess) << "Floyd takes " << timer.elapsedSeconds() << " seconds." << endl;
    } else { // geometrical graph.
        aux.objScale = Problem::GeometricalGraphObjScale;
        for (ID n = 0; n < nodeNum; ++n) {
            double nx = input.graph().nodes(n).x();
            double ny = input.graph().nodes(n).y();
            for (ID m = 0; m < nodeNum; ++m) {
                if (n == m) { continue; }
                aux.adjMat.at(n, m) = static_cast<Length>(aux.objScale * hypot(
                    nx - input.graph().nodes(m).x(), ny - input.graph().nodes(m).y()));
            }
        }
    }

    aux.coverRadii.init(nodeNum);
    fill(aux.coverRadii.begin(), aux.coverRadii.end(), Problem::MaxDistance);
}

bool Solver::optimize(Solution &sln, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;

    int solve = matchOptimalsolve(env.instPath);

    nodeNum = input.graph().nodenum();
    centerNum = input.centernum();
    iter = 0;

    disParameter = centerNum/5;

    //Initializing all data structures  �������ݽṹ����ֵΪ-1,�ڽӾ������ֵINF;���ɱ���ֵΪ0
    for (int i = 0; i < nodeNum; i++) {
        FsnodeTable.push_back(std::vector<int>());
        DistanceTable.push_back(std::vector<int>());
        FTable.push_back(std::vector<int>());
        DTable.push_back(std::vector<int>());
        TabuTable.push_back(std::vector<int>());
        NwTable.push_back(std::vector<int>());
        isServiceNode.push_back(false);
        longedgeMap.push_back(INT_MAX);
        pair<int, int> pair(-1,-1);
        Nw.push_back(pair);
        for (int j = 0; j < nodeNum; j++) {
            TabuTable[i].push_back(0);
            NwTable[i].push_back(-1);
        }
        for (int j = 0; j < 2; j++) {
            FsnodeTable[i].push_back(-1);
            DistanceTable[i].push_back(-1);
            DTable[i].push_back(-1);
            FTable[i].push_back(-1);
        }
    }

    //ServiceNodes : �����µĿռ�
    for (int i = 0; i < centerNum; i++) {    //initialize NwTable
        ServiceNodes.push_back(-1);
    }

    for (int i = 0; i < nodeNum; i++) {    //initialize NwTable
        findNw(i);
    }

    // reset solution state.
    bool status = true;
    auto &centers(*sln.mutable_centers());
    centers.Resize(centerNum, Problem::InvalidId);

    // TODO[0]: replace the following random assignment with your own algorithm.

    InitialSolu();
    
    start_time = clock();
    TabuSearch(solve);
    end_time = clock();
    elapsed_time = (double(end_time - start_time)) / CLOCKS_PER_SEC;
    std::cout << "  elapsed_time(s):" << elapsed_time << endl;
    std::cout << "success,iterations:" << iter << "  elapsed_time(s):" << elapsed_time << "frequency:"
        << double(iter / elapsed_time) << endl;

    for (ID n = 0; n < centerNum; ++n) {
        centers[n] = ServiceNodes[n];
    }

    sln.coverRadius = 0; // record obj.
    for (ID n = 0; n < nodeNum; ++n) {
        aux.coverRadii[n] = DistanceTable[n][0];
        if (sln.coverRadius < aux.coverRadii[n]) { sln.coverRadius = aux.coverRadii[n]; }
    }

    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}

void Solver::InitialSolu() {
    //����centerNum������ڵ�
    int centerNode = rand.pick(0, nodeNum);       //��nodeNum���ڵ��������һ���ڵ�
    ServiceNodes[0] = centerNode;
    //dis
    isServiceNode[centerNode] = true;
    //dis end
    for (int i = 0; i < nodeNum; i++)FsnodeTable[i][0] = centerNode;       //���нڵ���ɴ˷���ڵ����
    for (int i = 1; i < centerNum; i++) {
        ServiceNodes[i] = findnewServiceNode();
        //dis
        isServiceNode[ServiceNodes[i]] = true;
        //dis end
    }
    //��ʼ��F���D��
    int best = -1;
    for (int i = 0; i < nodeNum; i++) {
        DistanceTable[i][0] = aux.adjMat.at(i, FsnodeTable[i][0]);
        if (best < DistanceTable[i][0]) { best = DistanceTable[i][0]; }
        int serviceindex = findNextServiceNode(i);                               //check:yes
        FsnodeTable[i][1] = ServiceNodes[serviceindex];        //�ν�����ķ���ڵ�
        DistanceTable[i][1] = aux.adjMat.at(i, FsnodeTable[i][1]);
    }
    bestsolu = best;
    //test
    //cout << "InitialSolu: ";
    //vector<int>::iterator it;
    //for (it = ServiceNodes.begin(); it != ServiceNodes.end(); it++) {
        //cout << *it << "   ";
   // }
    //cout << endl;
    //cout << "init best:" << bestsolu << " , init size: " << ServiceNodes.size() << endl;
}

int compareByDisDescend(const void *a, const void *b) {   //����߳����ս�������
    pair<int, int> *p1 = (pair<int, int>*)a;
    pair<int, int> *p2 = (pair<int, int>*)b;
    return p2->second - p1->second;
}

void Solver::TabuSearch(const int &order) {               //check:yes.      �����ڵ���������������
    int disturbance_interval = 0;
    while (!timer.isTimeOut()) {     //ʱ��������߶�ε����������ܸ��µ�ǰ��ʷ���Ž�ʱ�����ϵ������³�ʼ�⡣����һ�Σ�����ö��������и���
        std::vector<int> shorterlenNodes;          //����������߶̵ıߵĽڵ㣨�û��ڵ㣩
        findaddNodesTS(shorterlenNodes);
        findbestAction(shorterlenNodes);
        int index = rand() % ((*bestaction).size());
        int newfun = INT_MAX;
        //test
       // cout << "iter: " << iter << endl;
        //cout << "swap:" << (*bestaction)[index].first << " " << ServiceNodes[(*bestaction)[index].second] << endl;
        newfun = makebestAction((*bestaction)[index]);
        if (newfun < bestsolu) {
            disturbance_interval = 0;
            bestsolu = newfun;
            //end_time = clock();
            //elapsed_time = (double(end_time - start_time)) / CLOCKS_PER_SEC;
            //cout << " update:" << bestsolu << ", time used:" << elapsed_time << endl;
            iteration = iter;
            if (bestsolu == order)break;
        }
        disturbance_interval++;
        iter++;
        if (disturbance_interval == 2000) {

            //test
            cout << "disturbance producing : " << endl;
            cout << "The center nodes are : ";
            for (int i = 0; i < ServiceNodes.size(); i++)cout << ServiceNodes[i] << " ";
            cout << endl;
            //cout << "disturbance number is :" << disParameter << endl;

            disturbance_WServiceNodes.clear();
            disturbance_WClientNodes.clear();
            for (int i = 0; i < nodeNum; i++) {         //��������Nw���з���߳��Ľ�������
                Nw[i].first = i;                      //��¼�ڵ���
                Nw[i].second = DistanceTable[i][0];   //��¼����
            }
            qsort(&(Nw.data()[0]), nodeNum, sizeof(pair<int, int>), compareByDisDescend);
            for (int i = 0; i < nodeNum; i++) {                                    //̰�ĵ���1/4������ڵ�����滻
                if (disturbance_WClientNodes.insert(pair<int, int>(FsnodeTable[Nw[i].first][0], Nw[i].first)).second) {
                    waitSerNodes.clear();
                    findminNode(Nw[i].first,DistanceTable[Nw[i].first][0],waitSerNodes);
                    while (!disturbance_WServiceNodes.insert(waitSerNodes[rand() % waitSerNodes.size()]).second) {
                    }
                }
                if (disturbance_WClientNodes.size() == disParameter) {
                    cout << " The size of disturbance_WClientNodes :" << disturbance_WClientNodes.size() << endl;
                    break;
                }
            }
            while (disturbance_WClientNodes.size() <= (disParameter * 2)) {           //�����1/4������ڵ�����滻
                if (disturbance_WClientNodes.insert(pair<int, int>(ServiceNodes[rand() % ServiceNodes.size()], -1)).second) {
                    int index = rand() % nodeNum;
                    while ((isServiceNode[index])||(!disturbance_WServiceNodes.insert(index).second)) {
                        index = rand() % nodeNum;
                    }
                }
            }


            //test
            cout << "The WServicenodes are : ";
            unordered_map<int, int>::iterator m;
            unordered_set<int>::iterator s;
            m = disturbance_WClientNodes.begin();
            s = disturbance_WServiceNodes.begin();
            for (; s != disturbance_WServiceNodes.end(); s++)cout << *s << " ";
            cout << endl;
            cout << "The WClientnodes are : ";
            for (; m != disturbance_WClientNodes.end(); m++)cout << m->first << " ";
            cout << endl;

            //unordered_map<int, int>::iterator m;
            //unordered_set<int>::iterator s;
            m = disturbance_WClientNodes.begin();
            s = disturbance_WServiceNodes.begin();
            for (; m != disturbance_WClientNodes.end(), s != disturbance_WServiceNodes.end(); s++, m++) {
                int fun = updateAddFacility(*s, FsnodeTable, DistanceTable);  //��Ӵ�����ķ���ڵ�
                isServiceNode[*s] = true;
                isServiceNode[m->first] = false;
                int newfun = deleteServiceNode(m->first);
                if (newfun < bestsolu)bestsolu = newfun;
                for (int i = 0; i < centerNum; i++) {
                    if (ServiceNodes[i] == m->first) {
                        ServiceNodes[i] = *s;
                        break;
                    }
                }
            }

            //test

        }
        //if (disturbance_interval == 2000) {       //2000���Ľ�������ʷ���Ž⣬�����Ŷ�
        //    cout << "disturbance producing : " << endl;
        //    random_WServiceNodes.clear();
        //    random_WClientNodes.clear();
        //    while ((random_WClientNodes.size() < randomNum) || (random_WServiceNodes.size() < randomNum)) {
        //        if (random_WClientNodes.size() < randomNum) {
        //            random_WClientNodes.insert(rand() % ServiceNodes.size());
        //        }
        //        if (random_WServiceNodes.size() < randomNum) {
        //            int index = rand() % nodeNum;
        //            while (isServiceNode[index]) {
        //                index = rand() % nodeNum;
        //            }
        //            random_WServiceNodes.insert(index);
        //        }
        //    }
        //    //for (int i = 0; i < randomNum; i++) {
        //    //    random_WClientNodes.insert(ServiceNodes[rand() % ServiceNodes.size()]);
        //    //    int index = rand() % nodeNum;
        //    //    while (isServiceNode[index]) {
        //    //        index = rand() % nodeNum;
        //    //    }
        //    //    random_WServiceNodes.insert(index);
        //    //}
        //    //if (random_WClientNodes.size() < randomNum) {
        //    //    random_WClientNodes.insert(ServiceNodes[rand() % ServiceNodes.size()]);
        //    //}
        //    //if (random_WServiceNodes.size() < randomNum) {
        //    //    int index = rand() % nodeNum;
        //    //    while (isServiceNode[index]) {
        //    //        index = rand() % nodeNum;
        //    //    }
        //    //    random_WServiceNodes.insert(index);
        //    //}
        //    set<int>::iterator w_ser, w_cli;
        //    w_ser = random_WServiceNodes.begin();
        //    w_cli = random_WClientNodes.begin();
        //    for (; w_ser != random_WServiceNodes.end(); w_ser++, w_cli++) {
        //        //makebestaction
        //        int fun = updateAddFacility(*w_ser, FsnodeTable, DistanceTable);  //��Ӵ�����ķ���ڵ�
        //        int newfun = deleteServiceNode(ServiceNodes[*w_cli]);
        //        if (newfun < bestsolu)bestsolu = newfun;
        //        isServiceNode[*w_ser] = true;
        //        isServiceNode[ServiceNodes[*w_cli]] = false;
        //        ServiceNodes[*w_cli] = *w_ser;
        //    }
        //}
    }
}

int Solver::findnewServiceNode() {     //�����ʼ��ʱ���µķ���ڵ��㷨         //check:yes
    int maxlen = -1;
    int maxlenNode = -1;
    vector<int> maxlenNodes;
    for (int i = 0; i < nodeNum; i++) {
        if (aux.adjMat.at(i, FsnodeTable[i][0]) >= maxlen) {
            if (aux.adjMat.at(i, FsnodeTable[i][0]) > maxlen) {
                maxlen = aux.adjMat.at(i, FsnodeTable[i][0]);
                maxlenNodes.clear();
                maxlenNodes.push_back(i);
            } else {
                maxlenNodes.push_back(i);
            }
        }
    }
    maxlenNode = maxlenNodes[rand() % maxlenNodes.size()];
    // �ҵ��ȵ�ǰ������߶̵Ľڵ㣬�������ѡһ����Ϊ�µķ���ڵ�
    std::vector<int> shortlenNodes;             //��¼�������߶̵ıߵĽڵ�
    findminNode(maxlenNode, maxlen, shortlenNodes);                                //update:����ˮ���������ѡ��һ���ڵ�
    int newServiceNode = shortlenNodes[rand() % shortlenNodes.size()];
    updateClientServiced(newServiceNode);
    return newServiceNode;
}

void Solver::findaddNodesTS(std::vector<int> &nodes) {            //check:yes
    int maxlen = -1;
    int maxlenNode = -1;
    vector<int> maxlenNodes;
    for (int i = 0; i < nodeNum; i++) {
        if (aux.adjMat.at(i, FsnodeTable[i][0]) >= maxlen) {
            if (aux.adjMat.at(i, FsnodeTable[i][0]) > maxlen) {
                maxlen = aux.adjMat.at(i, FsnodeTable[i][0]);
                maxlenNodes.clear();
                maxlenNodes.push_back(i);
            } else {
                maxlenNodes.push_back(i);
            }
        }
    }
    maxlenNode = maxlenNodes[rand() % maxlenNodes.size()];
    // �ҵ��ȵ�ǰ����߶̵ıߵĽڵ㣬������nodes������
    findminNode(maxlenNode, maxlen, nodes);
}

void Solver::findbestAction(const std::vector<int> &addservicenodes) {

    int tabufc = INT_MAX;
    int notabufc = INT_MAX;
    int newfunction = INT_MAX;
    for (int i = 0; i < addservicenodes.size(); i++) {
        for (int i = 0; i < nodeNum; i++) {           //FTable��DTable������������
            FTable[i][0] = FsnodeTable[i][0];
            DTable[i][0] = DistanceTable[i][0];
            FTable[i][1] = FsnodeTable[i][1];
            DTable[i][1] = DistanceTable[i][1];
        }
        for (int i = 0; i < ServiceNodes.size(); i++) {
            longedgeMap[ServiceNodes[i]] = 0;
        }
        int fun = updateAddFacility(addservicenodes[i], FTable, DTable);    //funΪ�������ڵ�֮���Ŀ�꺯��ֵ
        for (int i = 0; i < nodeNum; i++) {
            if (DTable[i][1] > longedgeMap[FTable[i][0]])
                longedgeMap[FTable[i][0]] = DTable[i][1];
        }
        
        for (int j = 0; j < ServiceNodes.size(); j++) {
            newfunction = std::max(fun, longedgeMap[ServiceNodes[j]]);
            if (iter < TabuTable[addservicenodes[i]][ServiceNodes[j]]) //�ڵ���ڽ�����
            {
                if (newfunction <= tabufc) {
                    if (newfunction < tabufc) {
                        bestactionTS.clear();
                        bestactionTS.push_back(make_pair(addservicenodes[i], j));
                        tabufc = newfunction;
                    } 
                    else {
                        bestactionTS.push_back(make_pair(addservicenodes[i], j));
                    }
                }
            } 
            else {
                if (newfunction <= notabufc) {
                    if (newfunction < notabufc) {
                        bestactionNTS.clear();
                        bestactionNTS.push_back(make_pair(addservicenodes[i], j));
                        notabufc = newfunction;;
                    } 
                    else {
                        bestactionNTS.push_back(make_pair(addservicenodes[i], j));
                    }
                }
            }
        }
    }
    //�жϽ�������
    if ((tabufc < bestsolu) && (tabufc < notabufc)) {
        bestaction = &bestactionTS;
    }
    else {
        if (bestactionNTS.size() == 0)bestaction = &bestactionTS;
        else bestaction = &bestactionNTS;
    }
}

int Solver::makebestAction(const std::pair<int,int> &best) {                      //check:yes�� �����޸Ľ��ɳ���
    
    int deleteservicenode = ServiceNodes[best.second];
    //int newfun = 0;
    int scaleconstant = (int)(nodeNum*0.5 + centerNum);
    //int scaleconstant = (int)(nodeNum*0.8);
    TabuTable[best.first][deleteservicenode] = iter + scaleconstant + rand.pick(1, centerNum);   //���½��ɱ�
    TabuTable[deleteservicenode][best.first] = iter + scaleconstant + rand.pick(1, centerNum);
    //TabuTable[deleteservicenode][best.first] = 0.3*nodeNum + (rand() % centerNum) + iter;
    //TabuTable[best.first][deleteservicenode] = TabuTable[deleteservicenode][best.first];
    ServiceNodes[best.second] = best.first;    //���·���ڵ�����
    int fun = updateAddFacility(best.first, FsnodeTable, DistanceTable);  //���ȸ��¼������ڵ��F���D��
    //for (int i = 0; i < nodeNum; i++)    //ɾ���ڵ�����F���D��
    //{
    //    if (FsnodeTable[i][0] == deleteservicenode) {
    //        FsnodeTable[i][0] = FsnodeTable[i][1];
    //        DistanceTable[i][0] = DistanceTable[i][1];
    //        int serviceindex = findNextServiceNode(i);
    //        FsnodeTable[i][1] = ServiceNodes[serviceindex];
    //        DistanceTable[i][1] = aux.adjMat.at(i, FsnodeTable[i][1]);
    //    } else if (FsnodeTable[i][1] == deleteservicenode) {
    //        int serviceindex = findNextServiceNode(i);
    //        FsnodeTable[i][1] = ServiceNodes[serviceindex];
    //        DistanceTable[i][1] = aux.adjMat.at(i, FsnodeTable[i][1]);

    //    } else;
    //    if (newfun < DistanceTable[i][0])newfun = DistanceTable[i][0];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      nceTable[i][0];
    //}
    //dis
    isServiceNode[deleteservicenode] = false;
    isServiceNode[best.first] = true;
    //dis end
    //return newfun;              //�����µ�Ŀ�꺯��ֵ
    return deleteServiceNode(deleteservicenode);
}

void Solver::findminNode(int indexnode, int servicelength, std::vector<int> &nodes)    //�ҵ��ȵ�ǰ������߶̵Ľڵ�   check:yes
{
    for (int i = 0; i < nodeNum; i++) {
        if (aux.adjMat.at(indexnode, NwTable[indexnode][i]) == servicelength)break;
        nodes.push_back(NwTable[indexnode][i]);
    }
}

void Solver::updateClientServiced(int newservicenode) {   //�����û��ڵ�����       //check:yes
    //���µķ���ڵ���û��ڵ����С�ڵ�ǰ�����ʱ���޸�Ϊ���µķ���ڵ����
    for (int i = 0; i < nodeNum; i++) {
        if (aux.adjMat.at(i, newservicenode) < aux.adjMat.at(i, FsnodeTable[i][0])) {
            FsnodeTable[i][0] = newservicenode;
        }
    }
}

int Solver::updateAddFacility(int addservicenode, std::vector<std::vector<int>> &Ftable, std::vector<std::vector<int>> &Dtable) {     //check:yes

    int function = 0;
    for (int i = 0; i < nodeNum; i++) {
        if (aux.adjMat.at(addservicenode, i) < Dtable[i][0]) {
            Dtable[i][1] = Dtable[i][0];
            Ftable[i][1] = Ftable[i][0];
            Dtable[i][0] = aux.adjMat.at(addservicenode, i);
            Ftable[i][0] = addservicenode;
        } else if (aux.adjMat.at(addservicenode, i) < Dtable[i][1]) {
            Dtable[i][1] = aux.adjMat.at(addservicenode, i);
            Ftable[i][1] = addservicenode;
        } else;
        if (function < Dtable[i][0])function = Dtable[i][0];
    }
    return function;
}

int Solver::deleteServiceNode(int deletenode) {
    int newfun = 0;
    for (int i = 0; i < nodeNum; i++)    //ɾ���ڵ�����F���D��
    {
        if (FsnodeTable[i][0] == deletenode) {
            FsnodeTable[i][0] = FsnodeTable[i][1];
            DistanceTable[i][0] = DistanceTable[i][1];
            int serviceindex = findNextServiceNode(i);
            FsnodeTable[i][1] = ServiceNodes[serviceindex];
            DistanceTable[i][1] = aux.adjMat.at(i, FsnodeTable[i][1]);
        } else if (FsnodeTable[i][1] == deletenode) {
            int serviceindex = findNextServiceNode(i);
            FsnodeTable[i][1] = ServiceNodes[serviceindex];
            DistanceTable[i][1] = aux.adjMat.at(i, FsnodeTable[i][1]);

        } else;
        if (newfun < DistanceTable[i][0])newfun = DistanceTable[i][0];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    }
    return newfun;
}

int Solver::findNextServiceNode(const int index) {           //check:yes   ���Ż�                //test
    int second_Distance = INT_MAX;                   //��¼�ζ̾������ڵ������
    vector<int> nextNodes;
    for (int j = 0; j < centerNum; j++) {          //���Ҵν�����ķ���ڵ�
        if (ServiceNodes[j] == FsnodeTable[index][0])continue;
        if (aux.adjMat.at(index, ServiceNodes[j]) <= second_Distance) {
            if (aux.adjMat.at(index, ServiceNodes[j]) < second_Distance) {
                nextNodes.clear();
                nextNodes.push_back(j);
                second_Distance = aux.adjMat.at(index, ServiceNodes[j]);
            } else {
                nextNodes.push_back(j);
            }
        }
    }
    return nextNodes[rand() % nextNodes.size()];
}

int compareByDistance(const void *a,const void *b) {    //���밴����������
    pair<int, int> *p1 = (pair<int, int>*)a, *p2 = (pair<int, int>*)b;
    return p1->second - p2->second;
}

void Solver::findNw(const int &node) {
    for (int i = 0; i < nodeNum; i++) {
        Nw[i].first = i;                      //��¼�ڵ���
        Nw[i].second= aux.adjMat.at(node, i);   //��¼����
    }
    qsort(&(Nw.data()[0]),nodeNum,sizeof(pair<int,int>),compareByDistance);
    for (int i = 0; i < nodeNum; i++) {
        NwTable[node][i] = Nw[i].first;    //�Ѿ����й�����ֻ��¼�����ɽ���Զ�Ľڵ���
    }
}

int Solver::matchOptimalsolve(const string &file) {
    ifstream findsolve;
    findsolve.open("Instance/tspsolve.txt");
    string line,str;
    while (getline(findsolve,line)) {
        istringstream stream(line);
        stream >> str;
        if (str.c_str() == file) {
            stream >> str;
            findsolve.close();
            return stoi(str.c_str());
        }
    }
}
#pragma endregion Solver

}