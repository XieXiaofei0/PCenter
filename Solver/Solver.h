////////////////////////////////
/// usage : 1.	the entry to the algorithm for the guillotine cut problem.
/// 
/// note  : 1.	
////////////////////////////////

#ifndef SMART_SZX_P_CENTER_SOLVER_H
#define SMART_SZX_P_CENTER_SOLVER_H


#include "Config.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <sstream>
#include <thread>

#include "Common.h"
#include "Utility.h"
#include "LogSwitch.h"
#include "Problem.h"

const int INF = 1000000000;

namespace szx {

struct BestAction {       //邻域动作定义
    int addServiceNode;     //要增加的服务节点
    int deleteSericeNodeIndex;    //要删除的服务节点的索引
    int NewFuntion;        //新的目标函数值：最长的服务边长度
    BestAction()
        : addServiceNode(-1)
        , deleteSericeNodeIndex(-1)
        , NewFuntion(INF) {
    }
};

class Solver {
    #pragma region Type
public:
    // commmand line interface.
    struct Cli {
        static constexpr int MaxArgLen = 256;
        static constexpr int MaxArgNum = 32;

        static String InstancePathOption() { return "-p"; }
        static String SolutionPathOption() { return "-o"; }
        static String RandSeedOption() { return "-s"; }
        static String TimeoutOption() { return "-t"; }
        static String MaxIterOption() { return "-i"; }
        static String JobNumOption() { return "-j"; }
        static String RunIdOption() { return "-rid"; }
        static String EnvironmentPathOption() { return "-env"; }
        static String ConfigPathOption() { return "-cfg"; }
        static String LogPathOption() { return "-log"; }

        static String AuthorNameSwitch() { return "-name"; }
        static String HelpSwitch() { return "-h"; }

        static String AuthorName() { return "szx"; }
        static String HelpInfo() {
            return "Pattern (args can be in any order):\n"
                "  exe (-p path) (-o path) [-s int] [-t seconds] [-name]\n"
                "      [-iter int] [-j int] [-id string] [-h]\n"
                "      [-env path] [-cfg path] [-log path]\n"
                "Switches:\n"
                "  -name  return the identifier of the authors.\n"
                "  -h     print help information.\n"
                "Options:\n"
                "  -p     input instance file path.\n"
                "  -o     output solution file path.\n"
                "  -s     rand seed for the solver.\n"
                "  -t     max running time of the solver.\n"
                "  -i     max iteration of the solver.\n"
                "  -j     max number of working solvers at the same time.\n"
                "  -rid   distinguish different runs in log file and output.\n"
                "  -env   environment file path.\n"
                "  -cfg   configuration file path.\n"
                "  -log   activate logging and specify log file path.\n"
                "Note:\n"
                "  0. in pattern, () is non-optional group, [] is optional group\n"
                "     when -env option is not given.\n"
                "  1. an environment file contains information of all options.\n"
                "     explicit options get higher priority than this.\n"
                "  2. reaching either timeout or iter will stop the solver.\n"
                "     if you specify neither of them, the solver will be running\n"
                "     for a long time. so you should set at least one of them.\n"
                "  3. the solver will still try to generate an initial solution\n"
                "     even if the timeout or max iteration is 0. but the solution\n"
                "     is not guaranteed to be feasible.\n";
        }

        // a dummy main function.
        static int run(int argc, char *argv[]);
    };

    // controls the I/O data format, exported contents and general usage of the solver.
    struct Configuration {
        enum Algorithm { Greedy, TreeSearch, DynamicProgramming, LocalSearch, Genetic, MathematicallProgramming };


        Configuration() {}

        void load(const String &filePath);
        void save(const String &filePath) const;


        String toBriefStr() const {
            String threadNum(std::to_string(threadNumPerWorker));
            std::ostringstream oss;
            oss << "alg=" << alg
                << ";job=" << threadNum;
            return oss.str();
        }


        Algorithm alg = Configuration::Algorithm::Greedy; // OPTIMIZE[szx][3]: make it a list to specify a series of algorithms to be used by each threads in sequence.
        int threadNumPerWorker = (std::min)(1, static_cast<int>(std::thread::hardware_concurrency()));
    };

    // describe the requirements to the input and output data interface.
    struct Environment {
        static constexpr int DefaultTimeout = (1 << 30);
        static constexpr int DefaultMaxIter = (1 << 30);
        static constexpr int DefaultJobNum = 0;
        // preserved time for IO in the total given time.
        static constexpr int SaveSolutionTimeInMillisecond = 1000;

        static constexpr Duration RapidModeTimeoutThreshold = 600 * static_cast<Duration>(Timer::MillisecondsPerSecond);

        static String DefaultInstanceDir() { return "Instance/"; }
        static String DefaultSolutionDir() { return "Solution/"; }
        static String DefaultVisualizationDir() { return "Visualization/"; }
        static String DefaultEnvPath() { return "env.csv"; }
        static String DefaultCfgPath() { return "cfg.csv"; }
        static String DefaultLogPath() { return "log.csv"; }

        Environment(const String &instancePath, const String &solutionPath,
            int randomSeed = Random::generateSeed(), double timeoutInSecond = DefaultTimeout,
            Iteration maxIteration = DefaultMaxIter, int jobNumber = DefaultJobNum, String runId = "",
            const String &cfgFilePath = DefaultCfgPath(), const String &logFilePath = DefaultLogPath())
            : instPath(instancePath), slnPath(solutionPath), randSeed(randomSeed),
            msTimeout(static_cast<Duration>(timeoutInSecond * Timer::MillisecondsPerSecond)), maxIter(maxIteration),
            jobNum(jobNumber), rid(runId), cfgPath(cfgFilePath), logPath(logFilePath), localTime(Timer::getTightLocalTime()) {}
        Environment() : Environment("", "") {}

        void load(const Map<String, char*> &optionMap);
        void load(const String &filePath);
        void loadWithoutCalibrate(const String &filePath);
        void save(const String &filePath) const;

        void calibrate(); // adjust job number and timeout to fit the platform.

        String solutionPathWithTime() const { return slnPath + "." + localTime; }

        String visualizPath() const { return DefaultVisualizationDir() + friendlyInstName() + "." + localTime + ".html"; }
        template<typename T>
        String visualizPath(const T &msg) const { return DefaultVisualizationDir() + friendlyInstName() + "." + localTime + "." + std::to_string(msg) + ".html"; }
        String friendlyInstName() const { // friendly to file system (without special char).
            auto pos = instPath.find_last_of('/');
            return (pos == String::npos) ? instPath : instPath.substr(pos + 1);
        }
        String friendlyLocalTime() const { // friendly to human.
            return localTime.substr(0, 4) + "-" + localTime.substr(4, 2) + "-" + localTime.substr(6, 2)
                + "_" + localTime.substr(8, 2) + ":" + localTime.substr(10, 2) + ":" + localTime.substr(12, 2);
        }

        // essential information.
        String instPath;
        String slnPath;
        int randSeed;
        Duration msTimeout;

        // optional information. highly recommended to set in benchmark.
        Iteration maxIter;
        int jobNum; // number of solvers working at the same time.
        String rid; // the id of each run.
        String cfgPath;
        String logPath;

        // auto-generated data.
        String localTime;
    };

    struct Solution : public Problem::Output { // cutting patterns.
        Solution(Solver *pSolver = nullptr) : solver(pSolver) {}

        Solver *solver;
    };
    #pragma endregion Type

    #pragma region Constant
public:
    #pragma endregion Constant

    #pragma region Constructor
public:
    Solver(const Problem::Input &inputData, const Environment &environment, const Configuration &config)
        : input(inputData), env(environment), cfg(config), rand(environment.randSeed),
        timer(std::chrono::milliseconds(environment.msTimeout)), iteration(1) {}
    #pragma endregion Constructor

    #pragma region Method
public:
    bool solve(); // return true if exit normally. solve by multiple workers together.
    bool check(Length &obj) const;
    void record() const; // save running log.

protected:
    void init();
    bool optimize(Solution &sln, ID workerId = 0); // optimize by a single worker

public:
    void InitialSolu();       //Initial solution of structure
    void TabuSearch(const int &order);   //如果超时，返回false；否则，返回true      
public:
    int findnewServiceNode();        //构造初始解时找新的服务节点算法
    void findaddNodesTS(std::vector<int> &nodes);     //局部搜索中查找可变为服务节点的用户节点数组,并返回当前的目标函数值
    void findbestAction(const std::vector<int> &addservicenodes, BestAction &bestmoveTS, BestAction &bestmoveNTS);    //局部搜索中找到最好的邻域动作
    int makebestAction(const BestAction &adddeletenodepair);       //局部搜索中做最好的邻域动作并返回新的目标函数值
public:
    void findminNode(int indexnode, int servicelength, std::vector<int> &nodes);    //找到比当前最大服务边短的节点
    void updateClientServiced(int newservicenode);      //更新用户节点数组
    int updateAddFacility(int addservicenode, std::vector<std::vector<int>> &Ftable, std::vector<std::vector<int>> &Dtable); //添加新的服务节点后更新F表和D表返回新的目标函数值
    int findNextServiceNode(const int deleteservicenode);   //根据删除的服务节点更新F表和D表
    #pragma endregion Method

    #pragma region Field
public:
    Problem::Input input;
    Problem::Output output;

    struct { // auxiliary data for solver.
        double objScale;

        Arr2D<Length> adjMat; // adjMat[i][j] is the distance of the edge which goes from i to j.
        Arr<Length> coverRadii; // coverRadii[n] is the length of its shortest serve arc.
    } aux;

    Environment env;
    Configuration cfg;

    Random rand; // all random number in Solver must be generated by this.
    Timer timer; // the solve() should return before it is timeout.
    Iteration iteration;

    int optimum_solution[40] = { 127,98,93,74,48,84,64,55,37,20,59,51,36,26,18,47,39,28,18,13,40,38,22,15,11,38,32,18,
13,9,30,29,15,11,30,27,15,29,23,13 };

    ID nodeNum;
    ID centerNum;
    ID bestsolu;      //keep the best solution of history
    ID iter;          //Iterative counting

    std::vector<int> ServiceNodes;               //Service nodes array
    std::vector<std::vector<int>> FsnodeTable;    //Nearest service node and secondary service node
    std::vector<std::vector<int>> DistanceTable;        //Distance of the nearest service node and secondary node
    std::vector<std::vector<int>> TabuTable;         //tabu table

    #pragma endregion Field
}; // Solver 

}


#endif // SMART_SZX_P_CENTER_SOLVER_H
