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
        constexpr bool IsUndirectedGraph = true;
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
    // by See
    int pos1 = env.instPath.find("d", 0);
    int pos2 = env.instPath.find(".", 0);
    cur_pmed_opt = optima_value[stoi(env.instPath.substr(pos1 + 1, pos2 - pos1 - 1)) - 1];
    // end See
}

// function definitions
void Solver::initializeGraph() { // ��ʼ��
    //��������0����������Ϊ+��
    solution.clear();
    fm_best = INT_MAX; // ��ʷ���Ŀ�꺯��ֵ
    fm_cur = INT_MAX; // ��ǰ��Ŀ�꺯��ֵ
    Sc_cur = INT_MAX; // p+1״̬��Ŀ�꺯��ֵ
    fm_expect = INT_MAX;
    rand_pair_happened = 1;
    // ȫ�ֱ���ҲҪ��ʼ��
    list_tabu = new int*[N_v];
    list_D = new int*[N_v];
    D = new int*[N_v];            //by Honesty:F,D����ʱ�ı�
    list_F = new int*[N_v];
    F = new int*[N_v];
    list_Nw = new int*[N_v];
    Nw = new pair<int, int>[N_v];
    Mf = new int[N_v];
    list_Index = new int[N_v];
    for (int i = 0; i < N_v; i++) {
        list_tabu[i] = new int[N_v];
        list_Nw[i] = new int[N_v];
        list_D[i] = new int[2];
        D[i] = new int[2];
        list_F[i] = new int[2];
        F[i] = new int[2];
    }
    for (int i = 0; i < N_v; i++) {
        for (int j = 0; j < N_v; j++) {
            if (i == j) {
                list_tabu[i][j] = -1;
                list_Nw[i][j] = 0;
            } else {
                if (i < j) { // ֻ��ֵ���Ͻ�
                    list_tabu[i][j] = -2;
                }
            }
        }
        for (int j = 0; j < 2; j++) {
            list_D[i][j] = INT_MAX;
            D[i][j] = INT_MAX;
            list_F[i][j] = -1;
            F[i][j] = -1;
        }
    }
    // calculate the Nw of each node.
    for (int i = 0; i < N_v; i++) {
        findNw(i);
    }

}

void Solver::addFacility(int f) {
    Sc_cur = 0;
    solution.insert(f);
    for (int v = 0; v < N_v; v++) {

        int d_fv = aux.adjMat[f][v];
        if (d_fv < list_D[v][0]) {
            list_D[v][1] = list_D[v][0];
            list_F[v][1] = list_F[v][0];
            list_D[v][0] = d_fv;
            list_F[v][0] = f;
        } else {
            if (d_fv < list_D[v][1]) {
                list_D[v][1] = d_fv;
                list_F[v][1] = f;
                //cout << "f1:" << v<<",dist:" << list_D[v][1] << endl;
            }
        }
        //cout << "v:" << v << ",dist:" << list_D[v][0] << "," << aux.adjMartrix[list_F[v][0]][v] <<",f0="<<list_F[v][0]
            //<<",  to 65:"<<aux.adjMartrix[65][v] << endl;
        if (list_D[v][0] > Sc_cur) {
            Sc_cur = list_D[v][0];
        }
    }
    //cout << "f______________________:" << f << endl;
    //cout << "add = " << Sc_cur << endl;
}

int compareByDistance(const void* a, const void* b) { // ��ֵĵط������ܷŵ�Solver����
    // ��������Nw����С����
    pair<int, int> *p1 = (pair<int, int>*)a, *p2 = (pair<int, int>*)b;
    return p1->second - p2->second;
}

void Solver::findNw(int w) {
    for (int j = 0; j < N_v; j++) {
        Nw[j].first = j; // first��¼�ڵ���
        Nw[j].second = aux.adjMat[w][j]; // second��¼�߳�
    }
    // �����ҳ���������±�k
    qsort(Nw, N_v, sizeof(pair<int, int>), compareByDistance);
    /*for (int i = 0; i < N_v; i++) {
        cout << "Nw w:" << w << ",1st:" << Nw[i].first << ",2nd:" << Nw[i].second << endl;
    }*/
    for (int j = 0; j < N_v; j++) {
        list_Nw[w][j] = Nw[j].first; // ��Ϊ�����Ѿ���С��������ֻ��¼�ڵ���
        /*if (w == 0) {
            cout << "N1[" << j << "]= " << list_Nw[w][j] << ",length=" << Nw[j].second << endl;
        }*/
    }
}

void Solver::findNwk(int s, int w) {
    // �����е�w��ʣ��ڵ���뵽Nw��
    // ���ԭ����
    Nwk.clear();
    for (int k = 0; k < N_v; k++) {
        //cout <<s<< "  �߳���" << Nw[k].second<<" ��һ��"<<Nw[k].first <<", ԭʼ����"<<list_D[w][0]<< endl;
        if (list_Nw[w][k] != s && aux.adjMat[w][s] > aux.adjMat[w][list_Nw[w][k]]) {
            Nwk.push_back(list_Nw[w][k]);
        } else break;
    }
}

void Solver::findLongestServeEdge() {
    int max = -1;
    longest_serve_edge.clear(); // ��֤ÿһ�����ǿռ�
    for (int i = 0; i < N_v; i++) {
        if (max < list_D[i][0]) {
            max = list_D[i][0];
            longest_serve_edge.clear(); // ֮ǰ��ȫ�����
            longest_serve_edge.push_back(make_pair(list_F[i][0], i)); // s , w
        } else {
            if (max == list_D[i][0]) {
                longest_serve_edge.push_back(make_pair(list_F[i][0], i));
            }
        }
    }
}

void Solver::initialzeASolution() {
    int random_first_facility = rand() % N_v; // ���ȡ�õ�һ�������
    pair<int, int> s_w; // ��¼������
    int f_; // ��¼������ڵ�
    int p_count = 1; // ��¼�Ѽ���ķ������Ŀ
    addFacility(random_first_facility);
    while (p_count != p) {
        findLongestServeEdge();
        s_w = longest_serve_edge[rand() % longest_serve_edge.size()];
        findNwk(s_w.first, s_w.second);
        f_ = Nwk[rand() % Nwk.size()];
        if (solution.find(f_) == solution.end()) {
            addFacility(f_);
            p_count++;
        }
    }
    fm_best = fm_cur = Sc_cur;
    /*for (int f : solution) {
        cout << ",sol: " << f;
    }
    cout << endl;*/
    //cout << "init best:" << fm_best << " , init size: " << solution.size() << endl;
}

int Solver::findMax() { // ����Ŀ�꺯��ֵ���������Ժ�findLongestServeEdge�Ż�
    int max = 0;
    for (int i = 0; i < N_v; i++) {
        if (max < list_D[i][0]) {
            max = list_D[i][0]; // ԭ���Ϸ���ڵ㵽����ı�=0�ǲ����ǵ�
        }
    }
    return max;
}

int Solver::isTaboo(int f, int i, int step) {
    // ����һ�Խ����ԣ��жϽ������
    if (f > i) {
        // �����ɱ������Ͻ�
        swap(f, i);
    }
    if (list_tabu[f][i] < step || list_tabu[f][i] == -2) {
        return 0;
    } else {
        return 1;
    }
}

void Solver::findNext(int v) {
    //ע��ִ�е��ú���ʱ��solution�����Ѿ��Ƴ��¼���ķ����
    int tmp_distance = INT_MAX;
    vector<int> second_closest_facility_set;
    second_closest_facility_set.clear();
    for (int f : solution) {
        if (tmp_distance >= aux.adjMat[v][f] && f != list_F[v][0]) { // �ҵ���ѵĴη������Ҳ��ú�Fv0��ͬ
            if (tmp_distance == aux.adjMat[v][f]) {
                second_closest_facility_set.push_back(f); // ͬ�Ⱥ�ѡ��
            } else {
                second_closest_facility_set.clear(); // ��ʤ��
                second_closest_facility_set.push_back(f);
                tmp_distance = aux.adjMat[v][f];
            }
        }
    }
    int random_next_server_index = rand() % (second_closest_facility_set.size()); // �Ӻ�ѡ�������ѡһ��������F D��
    list_F[v][1] = second_closest_facility_set[random_next_server_index];
    list_D[v][1] = tmp_distance;

}

void Solver::removeFacility(int f) {
    Sc_cur = 0;
    solution.erase(f);
    for (int v = 0; v < N_v; v++) {
        if (list_F[v][0] == f) {
            list_D[v][0] = list_D[v][1];
            list_F[v][0] = list_F[v][1];
            findNext(v);
        } else {
            if (list_F[v][1] == f) {
                findNext(v);
            }
        }
        if (list_D[v][0] > Sc_cur) {
            Sc_cur = list_D[v][0];
        }
    }
}

void Solver::removeFacility_(int f) {
    for (int i = 0; i < N_v; i++) {
        list_D[i][0] = D[i][0];
        list_D[i][1] = D[i][1];
        list_F[i][0] = F[i][0];
        list_F[i][1] = F[i][1];
    }
}

// Ѱ�ҽ����ԣ�����������ʽ
void Solver::findPair_(int s, int w, int step) {
    //set<int> solution_old(solution); // ��¼p+1֮ǰԭʼ��solution�������Ƚ�
    int* fast_solution_old = new int[p];
    int pc = 0;
    for (int f : solution) {
        fast_solution_old[pc] = f;
        pc++;
    }
    int fm_tabu = INT_MAX, fm_notabu = INT_MAX, fm_temp = INT_MAX;
    findNwk(s, w); // ��������
    set_L_notabu.clear();
    set_L_tabu.clear();
    //test
    //cout << "the will added nodes:" << endl;
    //for (int i : Nwk) {
    //    cout << i << " ";
    //}
    //cout << endl;
    //test end
    for (int i = 0; i < N_v; i++) { //��¼����������ָ�
        D[i][0] = list_D[i][0];
        D[i][1] = list_D[i][1];
        F[i][0] = list_F[i][0];
        F[i][1] = list_F[i][1];
    }
    for (int i : Nwk) {
        if (solution.find(i) == solution.end()) { // ȷ��i�������еķ�����У���������豸���豸����
            addFacility(i); // ��������˸÷����
            for (int pn = 0; pn < p; pn++) {
                Mf[fast_solution_old[pn]] = 0;
            }
            for (int v = 0; v < N_v; v++) {              //by Honesty:��¼ɾ��ĳ����ڵ������������߳���
                if (list_D[v][1] > Mf[list_F[v][0]]) {
                    Mf[list_F[v][0]] = list_D[v][1];
                }
            }
            for (int pn = 0; pn < p; pn++) { // ��һ��ͳ����ɾ�����Ǵ�����Mf������ѡ��ɾ��������ֲ���С����߳��ȵ�f��ͬʱ��Ϊ���ɺͷǽ�����������
                int f = fast_solution_old[pn];
                if (f != i) {
                    fm_temp = max(Sc_cur, Mf[f]);
                    if (isTaboo(f, i, step)) { // ��������и���
                        if (fm_temp < fm_tabu) {
                            fm_tabu = fm_temp;
                            set_L_tabu.clear();
                            set_L_tabu.push_back(make_pair(f, i));
                        } else {
                            if (fm_temp == fm_tabu) {
                                set_L_tabu.push_back(make_pair(f, i));
                            }
                        }
                    } else {
                        if (fm_temp < fm_notabu) {
                            fm_notabu = fm_temp;
                            set_L_notabu.clear();
                            set_L_notabu.push_back(make_pair(f, i));
                        } else {
                            if (fm_temp == fm_notabu) {
                                set_L_notabu.push_back(make_pair(f, i));
                            }
                        }
                    }
                }

            }
            solution.erase(i);
            removeFacility_(i); // ���ﲻ�ٸ���Sc_cur
        }
    }
    //test
    //cout << "bestsolu: " << fm_best << endl;
    //cout << "tabu_fm: " << fm_tabu << endl;
    //cout << "notabu_fm: " << fm_notabu << endl;
    //cout << "tabu_pair: " << set_L_tabu.size() << endl;
    //if (set_L_tabu.size() != 0) {
    //    for (pair<int, int> i : set_L_tabu) {
    //        cout << "  " << i.first << "  " << i.second << endl;
    //    }
    //    cout << endl;
    //}
    //cout << "notabu_pair: " << set_L_notabu.size() << endl;
    //if (set_L_notabu.size() != 0) {
    //    for (pair<int, int> i : set_L_notabu) {
    //        cout << "  " << i.first << "  " << i.second << endl;
    //    }
    //    cout << endl;
    //}
    //test end
    // �жϽ������
    if (fm_tabu < fm_best && fm_tabu < fm_notabu) {
        set_L = &set_L_tabu;
        //test
        //cout << "make tabu pair" << endl;
        //test end
    } else {
        if (set_L_notabu.size() == 0) {
            set_L = &set_L_tabu;
            //test
            //cout << "make tabu pair" << endl;
            //test end
        } else {
            set_L = &set_L_notabu;
        }

    }
}


void Solver::makeSwap(pair<int, int> s_w, int step, Solution &sln) {
    // ���½��ɱ�F D ��solution,Sc_best��
    int fm_notequal = fm_cur;
    if (s_w.first < s_w.second) {
        list_tabu[s_w.first][s_w.second] = 0.3*N_v + (rand() % p) + step;
    } else {
        list_tabu[s_w.second][s_w.first] = 0.3*N_v + (rand() % p) + step;
    }
    addFacility(s_w.second);
    removeFacility(s_w.first);
    fm_cur = Sc_cur;
    //test
    //cout << "swap: " << s_w.second << " " << s_w.first << endl;
    if (fm_cur < fm_best) {
        fm_best = fm_cur;
        end_time = clock();
        elapsed_time = (double(end_time - start_time)) / CLOCKS_PER_SEC;
        cout << " update:" << fm_best << ", time used:" << elapsed_time << endl;
        sln.clear_centers();
        for (int f : solution) {
            sln.add_centers(f);
        }
    }
}

int Solver::evaluateMf(int f_mf) {
    for (int f : solution) {
        Mf[f] = 0;
    }
    for (int v = 0; v < N_v; v++) {
        if (list_D[v][1] > Mf[list_F[v][0]]) {
            Mf[list_F[v][0]] = list_D[v][1];
        }
    }
    return Mf[f_mf];
}

bool Solver::optimize(Solution &sln, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;
    bool status = true;
    sln.coverRadius = 0;
    // copy the values of this program's variables to my own ones
    N_v = input.graph().nodenum();

    p = input.centernum();
    initializeGraph();
    pair<int, int> s_w;
    int random_swap_index = -1;
    int step = 0;
    initialzeASolution();
    //test
    cout << "Intialsolu: ";
    for (int f : solution) {
        cout << f << " ";
    }
    cout << endl;
    //test end
    start_time = clock();
    // TODO[0]: replace the following random assignment with your own algorithm.

    while (!timer.isTimeOut()) {
        findLongestServeEdge();
        //test
        //cout << "iter: " << step << endl;
        //cout << "the longest service_edges: " << endl;
        //for (int i = 0; i < longest_serve_edge.size(); i++)
        //    cout << "  " << i << ": " << longest_serve_edge[i].first << " " << longest_serve_edge[i].second << endl;
        //cout << endl;
        //test end
        s_w = longest_serve_edge[rand() % (longest_serve_edge.size())];
        //test
        //cout << "the longest edge_pair : " << s_w.first << " " << s_w.second << endl;
        //test end
        findPair_(s_w.first, s_w.second, step);
        random_swap_index = rand() % ((*set_L).size());
        //test
        //cout << "make pair_index��" << random_swap_index << endl;
        //cout << "the true swap pair is : " << (*set_L)[random_swap_index].first << " " << (*set_L)[random_swap_index].second << endl << endl << endl;
        //test end
        makeSwap((*set_L)[random_swap_index], step, sln);
        if (fm_best == cur_pmed_opt) {
            break;
        }
        //test
        //if (step == 10)break;
        //test end
        step++;
    }
    end_time = clock();
    elapsed_time = (double(end_time - start_time)) / CLOCKS_PER_SEC;
    cout << "the opt= " << fm_best << endl;
    std::cout << "success,iterations:" << step << "  elapsed_time(s):" << elapsed_time << "frequency:"
        << double(step / elapsed_time) << endl;
    sln.coverRadius = fm_best; // ���з�����е����ֵ������������
    Log(LogSwitch::Szx::Framework) << " worker" << workerId << " ends." << endl;
    return status;
}
#pragma endregion Solver

}
