using Python.Runtime;
using Classes;

namespace Q2
{
    public class Program
    {
        static void Main(string[] args)
        {
            int n = 9;
            double gamma = 1;
            double epsilon = 1e-15;

            List<Coordinate> obstacles = new List<Coordinate>
            {
                new Coordinate(3, 1),
                new Coordinate(3, 2),
                new Coordinate(3, 3),
                new Coordinate(2, 3),
                new Coordinate(1, 3),
                new Coordinate(5, 5),
                new Coordinate(5, 6),
                new Coordinate(5, 7),
                new Coordinate(5, 8),
                new Coordinate(6, 5),
                new Coordinate(7, 5),
                new Coordinate(8, 5)
            };

            Coordinate PortalIn = new Coordinate(2, 2);
            Coordinate PortalOut = new Coordinate(6, 6);

            Coordinate Goal = new Coordinate(8, 8);

            Environment env = new Environment(n, obstacles, PortalIn, PortalOut, Goal);
            
            MDP mdp = new MDP(env.States, env.Actions);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    
                    Coordinate current = new Coordinate(i, j);
                    

                    if (!env.Obstacles.Keys.ToArray().Contains(current) &&
                        !env.PortalIn.Keys.ToArray().Contains(current))
                    {
                        Coordinate LeftTile = new Coordinate(current.X - 1, current.Y);
                        Coordinate RightTile = new Coordinate(current.X + 1, current.Y);
                        Coordinate UpTile = new Coordinate(current.X, current.Y + 1);
                        Coordinate DownTile = new Coordinate(current.X, current.Y - 1);    
                        
                        var transitionCurrentLeft = CreateTransactionList(env, current, LeftTile);
                        var transitionCurrentRight = CreateTransactionList(env, current, RightTile);
                        var transitionCurrentUp = CreateTransactionList(env, current, UpTile);
                        var transitionCurrentDown = CreateTransactionList(env, current, DownTile);

                        if (env.Walkable.Keys.ToArray().Contains(current))
                        {
                            mdp.AddStateAction(env.Walkable[current], env.Actions.Left, transitionCurrentLeft);
                            mdp.AddStateAction(env.Walkable[current], env.Actions.Right, transitionCurrentRight);
                            mdp.AddStateAction(env.Walkable[current], env.Actions.Up, transitionCurrentUp);
                            mdp.AddStateAction(env.Walkable[current], env.Actions.Down, transitionCurrentDown);
                        }
                        else if (env.PortalOut.Keys.ToArray().Contains(current))
                        {
                            
                            mdp.AddStateAction(env.PortalOut[current], env.Actions.Left, transitionCurrentLeft);
                            mdp.AddStateAction(env.PortalOut[current], env.Actions.Right, transitionCurrentRight);
                            mdp.AddStateAction(env.PortalOut[current], env.Actions.Up, transitionCurrentUp);
                            mdp.AddStateAction(env.PortalOut[current], env.Actions.Down, transitionCurrentDown);
                        }
                        else if (env.Goal.Keys.ToArray().Contains(current))
                        {
                            mdp.AddStateAction(env.Goal[current], env.Actions.Left, transitionCurrentLeft);
                            mdp.AddStateAction(env.Goal[current], env.Actions.Right, transitionCurrentRight);
                            mdp.AddStateAction(env.Goal[current], env.Actions.Up, transitionCurrentUp);
                            mdp.AddStateAction(env.Goal[current], env.Actions.Down, transitionCurrentDown);
                        }
                    }
                }
            }

            Dictionary<State, double> vValueIter = new Dictionary<State, double>();
            Dictionary<State, double> vPolicyIter = new Dictionary<State, double>();
            Dictionary<State, Classes.Action> pValueIter = new Dictionary<State, Classes.Action>();
            Dictionary<State, Classes.Action> pPolicyIter = new Dictionary<State, Classes.Action>();

            mdp.PolicyIteration(out vPolicyIter, out pPolicyIter, gamma, epsilon);

            mdp.ValueIteration(out vValueIter, out pValueIter, gamma, epsilon);

            vValueIter[env.PortalIn.Values.ToArray()[0]] = vValueIter[env.PortalOut.Values.ToArray()[0]];
            vPolicyIter[env.PortalIn.Values.ToArray()[0]] = vPolicyIter[env.PortalOut.Values.ToArray()[0]];
            RunScript("main", mdp.States, vValueIter, vPolicyIter, pValueIter, pPolicyIter);
        }

        static List<Transition> CreateTransactionList(Environment env, Coordinate currentTile, Coordinate NextTile)
        {
            var transitionsCurrentNext = new List<Transition>();

            // Next Tile is Obstacle or Boundary
            if (NextTile.X < 0 || NextTile.Y < 0 || NextTile.X >= env.n || NextTile.Y >= env.n || env.Obstacles.Keys.Contains(NextTile))
            {
                State next;
                           
                if (env.Goal.Keys.Contains(currentTile))
                {
                    next = env.Goal[currentTile];
                }
                else if (env.Walkable.Keys.Contains(currentTile))
                {
                    next = env.Walkable[currentTile];
                }
                else
                {
                    next = env.PortalOut[currentTile];
                }
                Transition nextTransition  = new Transition(next, 0.25);
                transitionsCurrentNext.Add(nextTransition);
            }
            // Next Tile is portalIn
            else if (env.PortalIn.Keys.Contains(NextTile))
            {
                Coordinate portalOut = env.PortalOut.Keys.ToArray()[0];
                State next = env.PortalOut[portalOut];
                Transition nextTransition = new Transition(next, 0.25);
                transitionsCurrentNext.Add(nextTransition);
            }
            // Next Tile is walkable
            else if (env.Walkable.Keys.Contains(NextTile))
            {
                State next = env.Walkable[NextTile];
                Transition nextTransition = new Transition(next, 0.25);
                transitionsCurrentNext.Add(nextTransition);
            }
            // Next Tile is Goal
            else if (env.Goal.Keys.Contains(NextTile))
            {
                State next = env.Goal[NextTile];
                Transition nextTransition = new Transition(next, 0.25);
                transitionsCurrentNext.Add(nextTransition);
            }

            return transitionsCurrentNext;
        }

        static void RunScript(string scriptName,List<State> states, Dictionary<State,
                              double> vValueIter, Dictionary<State, double> vPolicyIter,
                              Dictionary<State, Classes.Action> pValueIter,
                              Dictionary<State, Classes.Action> pPolicyIter)
        {
            // Here the python dll or so file according to os and environment
            Runtime.PythonDLL = @"/home/shir0/miniconda3/envs/system/lib/libpython3.10.so";
            PythonEngine.Initialize();

            using(Py.GIL())
            {
                AddCurrentDirectoryToPythonSystem();

                var PythonScript = Py.Import(scriptName);
                var statesPython = states.Select(state => state.Name).ToList().ToPython();

                dynamic vValueIterPython = new PyDict();
                dynamic vPolicyIterPython = new PyDict();

                foreach (var kvp in vValueIter)
                {
                    vValueIterPython[kvp.Key.Name] = kvp.Value.ToPython();
                }

                foreach (var kvp in vPolicyIter)
                {
                    vPolicyIterPython[kvp.Key.Name] = kvp.Value.ToPython();
                }

                dynamic pValueIterPython = new PyDict();
                dynamic pPolicyIterPython = new PyDict();

                foreach (var kvp in pValueIter)
                {
                    pValueIterPython[kvp.Key.Name] = kvp.Value.Name.ToPython();
                }

                foreach (var kvp in pPolicyIter)
                {
                    pPolicyIterPython[kvp.Key.Name] = kvp.Value.Name.ToPython();
                }

                PythonScript.InvokeMethod("plot_valueIteration", new PyObject[] {statesPython, vValueIterPython, pValueIterPython});
                PythonScript.InvokeMethod("plot_policyIteration", new PyObject[] {statesPython, vPolicyIterPython, pPolicyIterPython});
            }
            ClosePythonEngine();
        }

        static void AddCurrentDirectoryToPythonSystem()
        {

                dynamic sys = Py.Import("sys");
                sys.path.append(".");
        }

        static void ClosePythonEngine()
        {
            AppContext.SetSwitch("System.Runtime.Serialization.EnableUnsafeBinaryFormatterSerialization", true);
            PythonEngine.Shutdown();
            AppContext.SetSwitch("System.Runtime.Serialization.EnableUnsafeBinaryFormatterSerialization", false);
        }
    }
}