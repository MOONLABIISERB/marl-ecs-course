using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Q2.Environment;
using Q2.Extension;

namespace Q2.RL
{
    public class Reinforcement_Learning
    {
        GameWindow Gamewindow;
        Level Level;
        State[] States;
        Environment.Action[] actions = {Environment.Action.Up, Environment.Action.Down, Environment.Action.Left, Environment.Action.Right};
        public double MovementReward { get; set; }
        public double CompletionReward { get; set; }
        static Random random = new Random(Seed:20031);
        
        public Dictionary<Point[], double> OptimalValue { get; set; }

        public Reinforcement_Learning(GameWindow window, Level level, double completionReward = 100, double movementReward = -1)
        {
            Gamewindow = window;
            Level = level;
            CompletionReward = completionReward;
            MovementReward = movementReward;
        }

        public void ValueIteration(out Dictionary<State, double> value, out Dictionary<State, Environment.Action> policy, double gamma, int numIter)
        {
            int current = 0;
            // Initialize Policies and values
            InitializePolicyAndValue(out policy,out value);

            while (current < numIter)
            {
                Dictionary<State, double> VNew = new Dictionary<State, double>();
                System.Console.WriteLine($"Iteration number:{current + 1}");
                foreach (var state in States)
                {
                    double maxValue = double.NegativeInfinity;
                    Environment.Action bestAction = Environment.Action.Right;
                    foreach (Environment.Action action in actions)
                    {
                        ResetLevel(state);
                        
                        Level.MovePlayer(action);
                        Point[] boxesPos = new Point[Level.Boxes.Length];
                        for (int i = 0; i < boxesPos.Length; i++)
                        {
                            boxesPos[i] = Level.Boxes[i].Position;
                        }
                        State newState = new State(Level.Player.Position, boxesPos);

                        double actionValue = MovementReward;
                        if (Level.LevelComplete())
                        {
                            actionValue = CompletionReward;
                        }

                        actionValue += gamma * value[newState];

                        if (actionValue > maxValue)
                        {
                            maxValue = actionValue;
                            bestAction = action;
                        }
                    }
                    
                    VNew[state] = maxValue;
                    policy[state] = bestAction;
                }
                value = VNew;
                current++;
            }
        }

        public void MCESFirstVisit(out Dictionary<State, Environment.Action> policy, out Dictionary<StateAction, double> Qvalue, double gamma, int numEpisodes, int maxSteps)
        {
            // Initialization
            States = GenerateStates();
            policy = new Dictionary<State, Environment.Action>();
            Qvalue = new();
            Dictionary<StateAction, List<double>> Returns = new();
            foreach (State state in States)
            {
                foreach (Environment.Action action in actions)
                {
                    Qvalue[new(state, action)] = double.NegativeInfinity; // Initializing QValue to -infinity
                }
                policy[state] = actions[random.Next(actions.Length)]; // Initializing with random actions
            }
            Console.WriteLine("Initiation Done!");

            // Loop For episodes
            for (int episode = 0; episode < numEpisodes; episode++)
            {   
                State state = States[random.Next(States.Length)];
                Environment.Action action = actions[random.Next(actions.Length)];

                HashSet<StateAction> stateAction = new();   
                List<EpisodeStep> episodeSteps = GenerateEpisode(state, action, policy, maxSteps);
                
                double G = 0;

                int t = episodeSteps.Count - 1;

                do 
                {
                    var stateActionPair = new StateAction(episodeSteps[t].State, episodeSteps[t].Action);
                    if (t == episodeSteps.Count - 1)
                    {
                        G = episodeSteps[t].Reward;
                    }
                    else
                    {
                        if (stateActionPair.State.PlayerPosition == new Point(4,3))
                        {

                        }
                        G = gamma * G + episodeSteps[t + 1].Reward;
                    }
                    if (stateAction.Add(stateActionPair))
                    {
                        if (!Returns.ContainsKey(stateActionPair))
                        {
                            Returns[stateActionPair] = new List<double>();
                        }
                        Returns[stateActionPair].Add(G);
                        Qvalue[stateActionPair] = General.Average(Returns[stateActionPair]);
                    }
                    t--;
                } while (t >= 0);


                PolicyUpdate(ref Qvalue, ref policy);
                if ((episode + 1) % (numEpisodes / 10) == 0)
                    Console.WriteLine($"Episode {episode + 1} of {numEpisodes} done");
            }   
            
              
        }

        public void PolicyUpdate(ref Dictionary<StateAction, double> QValue, ref Dictionary<State, Environment.Action> policy)
        {
            var actions = Enum.GetValues(typeof(Environment.Action));
            foreach (State state in States)
            {
                double bestQ = double.NegativeInfinity;
                Environment.Action bestAction = Environment.Action.Up;
                foreach (Environment.Action action in actions)
                {
                    var stateActionPairEval = new StateAction(state, action);
                    if (QValue[stateActionPairEval] > bestQ)
                    {
                        bestQ = QValue[stateActionPairEval];
                        bestAction = stateActionPairEval.Action;                                
                    }
                }
                policy[state] = bestAction;
            }
        }


        public void ResetLevel(State state)
        {
            Gamewindow.InitializeLevel();
            Level.Player.Position = state.PlayerPosition;

            for (int i = 0; i < Level.Boxes.Length; i++)
            {
                Level.Boxes[i].Position = state.BoxesPosition[i];
            }    
        }
        public void MCESEveryVisit(out Dictionary<State, Environment.Action> policy, 
                                out Dictionary<StateAction, double> Qvalue, 
                                double gamma, int numEpisodes, int maxSteps)
        {

            // Initialize
            States = GenerateStates();
            policy = new Dictionary<State, Environment.Action>();
            Qvalue = new();
            Dictionary<StateAction, List<double>> Returns = new();
            foreach (State state in States)
            {
                foreach (Environment.Action action in actions)
                {
                    Qvalue[new(state, action)] = double.NegativeInfinity; // Initializing QValue to -infinity
                }
                policy[state] = actions[random.Next(actions.Length)]; // Initializing with random actions
            }


            Console.WriteLine("Initiation Done!");

            // Loop For episodes
            for (int episode = 0; episode < numEpisodes; episode++)
            {   
                State state = States[random.Next(States.Length)];
                Environment.Action action = actions[random.Next(actions.Length)];   
                List<EpisodeStep> episodeSteps = GenerateEpisode(state, action, policy, maxSteps);
                
                double G = 0;

                int t = episodeSteps.Count - 1;

                do 
                {
                    var stateActionPair = new StateAction(episodeSteps[t].State, episodeSteps[t].Action);
                    if (t == episodeSteps.Count - 1)
                    {
                        G = episodeSteps[t].Reward;
                    }
                    else
                    {
                        if (stateActionPair.State.PlayerPosition == new Point(4,3))
                        {

                        }
                        G = gamma * G + episodeSteps[t + 1].Reward;
                    }
                    if (!Returns.ContainsKey(stateActionPair))
                    {
                        Returns[stateActionPair] = new List<double>();
                    }
                    Returns[stateActionPair].Add(G);
                    Qvalue[stateActionPair] = General.Average(Returns[stateActionPair]);
                    t--;
                } while (t >= 0);


                PolicyUpdate(ref Qvalue, ref policy);
                if ((episode + 1) % (numEpisodes / 10) == 0)
                    Console.WriteLine($"Episode {episode + 1} of {numEpisodes} done");
            }
        }

        private List<EpisodeStep> GenerateEpisode(State initState, Environment.Action initAction, Dictionary<State, Environment.Action> policy, int maxSteps)
        {
            List<EpisodeStep> episode = new();
            State state = initState;
            Environment.Action action = initAction;
            ResetLevel(state);

            int step = 0;
            while (step < maxSteps)
            {   
                Level.MovePlayer(action);
                Point[] boxesPos = new Point[Level.Boxes.Length];
                for (int i = 0; i < boxesPos.Length; i++)
                {
                    boxesPos[i] = Level.Boxes[i].Position;
                }
                double reward = MovementReward;
                if (Level.LevelComplete())
                {
                    reward = CompletionReward;
                    episode.Add(new EpisodeStep(state, action, reward));
                    break;
                }
                episode.Add(new EpisodeStep(state, action, reward));
                state = new State(Level.Player.Position, boxesPos);
                action = policy[state];
                step++;
            }

            return episode;
        }

        
        private void InitializePolicyAndValue(out Dictionary<State, Environment.Action> policy, out Dictionary<State, double> value)
        {
            // Initialize Policies and values
            policy = new Dictionary<State, Environment.Action>();
            value = new Dictionary<State, double>();
            States = GenerateStates();
            foreach (var state in States)
            {
                policy[state] = Environment.Action.Right;
                value[state] = 0;
            }
        }

        public State[] GenerateStates()
        {
            var positions = GetAllPositions();
            var states = new List<State>();
            // Generate all permutations for player + boxes
            var permutations = GetPermutations(positions, 1 + Level.Boxes.Length);
            foreach (var perm in permutations)
            {
                var state = new State(perm);
                bool walkable = true;
                // if (HasDuplicatePoints(perm))
                //     walkable = false;
                foreach (var point in perm)
                {
                    if (Level.Map[point.X, point.Y].Type == Extension.GameObjectType.Wall)
                        walkable = false;   
                }
                if (walkable)
                {
                    states.Add(state);
                }
            }

            return states.ToArray();
        }

        private List<Point> GetAllPositions()
        {
            var positions = new List<Point>();
            int rows = Level.Rows;
            int cols = Level.Columns;

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    positions.Add(Level.Map[x,y].Position);
                }
            }

            return positions;
        }

        private IEnumerable<Point[]> GetPermutations(List<Point> list, int length)
        {
            if (length == 1)
                return list.Select(t => new Point[] { t });

            return GetPermutations(list, length - 1)
                .SelectMany(t => list.Where(e => !t.Contains(e)),
                            (t1, t2) => t1.Concat(new[] { t2 }).ToArray());
        }

        public bool HasDuplicatePoints(Point[] points)
        {
            HashSet<Point> seenPoints = new HashSet<Point>();
            foreach (var point in points)
            {
                if (!seenPoints.Add(point))
                {
                    return true; // A duplicate point was found
                }
            }
            return false; // No duplicates
        }
    }
}