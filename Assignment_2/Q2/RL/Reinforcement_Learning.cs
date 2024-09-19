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
        static Random random = new Random(Seed:1);
        
        public Dictionary<Point[], double> OptimalValue { get; set; }

        public Reinforcement_Learning(GameWindow window, Level level, double gamma = 0.9, double epsilon = 1e-6)
        {
            Gamewindow = window;
            Level = level;
        }

        public void ValueIteration(out Dictionary<State, double> value, out Dictionary<State, Environment.Action> policy, double gamma, int numIter)
        {
            // bool isConverged = false;
            int current = 0;
            // Initialize Policies and values
            InitializePolicyAndValue(out policy,out value);
            while (current < numIter)
            {
                Dictionary<State, double> VNew = new Dictionary<State, double>();
                System.Console.WriteLine($"Iteration number:{current}");
                foreach (var state in States)
                {
                    double maxValue = double.NegativeInfinity;
                    Environment.Action bestAction = Environment.Action.Right;
                    foreach (Environment.Action action in Enum.GetValues(typeof(Environment.Action)))
                    {
                        Gamewindow.InitializeLevel();
                        Level.Player.Position = state.PlayerPosition;

                        for (int i = 0; i < Level.Boxes.Length; i++)
                        {
                            Level.Boxes[i].Position = state.BoxesPosition[i];
                        }
                        Level.MovePlayer(action);
                        Point[] boxesPos = new Point[Level.Boxes.Length];
                        for (int i = 0; i < boxesPos.Length; i++)
                        {
                            boxesPos[i] = Level.Boxes[i].Position;
                        }
                        State newState = new State(Level.Player.Position, boxesPos);

                        double actionValue = -1;
                        if (Level.LevelComplete())
                        {
                            actionValue = 0;
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
                    Qvalue[new(state, action)] = double.NegativeInfinity;
                }
                policy[state] = actions[random.Next(actions.Length)];
            }

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
                        Qvalue[stateActionPair] = Average(Returns[stateActionPair]);
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

public void MCESEveryVisit(out Dictionary<State, Environment.Action> policy, 
                           out Dictionary<StateAction, double> QValue, 
                           double gamma, int numIter, int maxSteps)
{
    States = GenerateStates();
    QValue = new Dictionary<StateAction, double>();
    policy = new Dictionary<State, Environment.Action>();
    Dictionary<StateAction, List<double>> returns = new Dictionary<StateAction, List<double>>();

    // Initialize Q-values and returns
    foreach (State state in States)
    {
        foreach (Environment.Action action in Enum.GetValues(typeof(Environment.Action)))
        {
            var stateActionPair = new StateAction(state, action);
            QValue[stateActionPair] = 0.0;
            returns[stateActionPair] = new List<double>();
        }
        policy[state] = Environment.Action.Up; // Default action
    }

    for (int k = 0; k < numIter; k++)
    {
        foreach (var state in States)
        {
            foreach (Environment.Action action in Enum.GetValues(typeof(Environment.Action)))
            {
                // Initialize the level with the current state
                Gamewindow.InitializeLevel();
                Level.Player.Position = state.PlayerPosition;
                for (int i = 0; i < Level.Boxes.Length; i++)
                {
                    Level.Boxes[i].Position = state.BoxesPosition[i];
                }

                double G = 0;
                int steps = 0;
                var stateActionPair = new StateAction(state, action);
                List<double> rewards = new List<double>();

                // Simulate the episode
                while (steps < maxSteps)
                {
                    Level.MovePlayer(action);
                    double Reward = -1; // Default negative reward

                    // Check if the level is complete (terminal state)
                    if (Level.LevelComplete())
                    {
                        Reward = 0; // Reward for completing the level
                        rewards.Add(Reward);
                        break; // End episode
                    }

                    rewards.Add(Reward);
                    steps++;
                }

                // Calculate the return G for this episode (discounted sum of rewards)
                G = rewards.Reverse<double>().Aggregate(0.0, (acc, reward) => gamma * acc + reward);

                // Debugging: Print rewards and return
                // Console.WriteLine($"Iteration {k}, State: {state}, Action: {action}, Rewards: {string.Join(", ", rewards)}, Return G: {G}");

                // Update returns and Q-value
                returns[stateActionPair].Add(G);
                QValue[stateActionPair] = returns[stateActionPair].Average();

                // Debugging: Print updated Q-value
                // Console.WriteLine($"Updated Q-value for State: {state}, Action: {action}: {QValue[stateActionPair]}");
            }
        }

        // Policy update: Select the action with the highest Q-value for each state
        foreach (var state in States)
        {
            double bestQ = double.NegativeInfinity;
            Environment.Action bestAction = Environment.Action.Up; // Default action

            foreach (Environment.Action action in Enum.GetValues(typeof(Environment.Action)))
            {
                var stateActionPair = new StateAction(state, action);
                if (QValue[stateActionPair] > bestQ)
                {
                    bestQ = QValue[stateActionPair];
                    bestAction = action;
                }
            }

            // Update the policy
            policy[state] = bestAction;

            // Debugging: Print the selected policy for this state
            // Console.WriteLine($"Updated policy for State: {state} -> Best Action: {bestAction}");
        }
    }
}

        private List<EpisodeStep> GenerateEpisode(State initState, Environment.Action initAction, Dictionary<State, Environment.Action> policy, int maxSteps)
        {
            
            List<EpisodeStep> episode = new();
            State state = initState;
            Environment.Action action = initAction;
            Gamewindow.InitializeLevel();
            Level.Player.Position = state.PlayerPosition;

            for (int i = 0; i < Level.Boxes.Length; i++)
            {
                Level.Boxes[i].Position = state.BoxesPosition[i];
            }
            int step = 0;
            while (step < maxSteps)
            {   
                Level.MovePlayer(action);
                Point[] boxesPos = new Point[Level.Boxes.Length];
                for (int i = 0; i < boxesPos.Length; i++)
                {
                    boxesPos[i] = Level.Boxes[i].Position;
                }
                double reward = -1;
                if (Level.LevelComplete())
                {
                    reward = 1000;
                    episode.Add(new EpisodeStep(state, action, reward));
                    break;
                }
                // if (Level.LevelUnsolvable())
                // {
                //     reward = -1 * maxSteps;
                //     episode.Add(new EpisodeStep(state, action, reward));
                //     break;
                // }
                // if (episode.Count > 0)
                // {
                //     var prevState = episode[episode.Count - 1].State;
                //     if (prevState == state)
                //     {
                //         reward = -1 * maxSteps;
                //         episode.Add(new EpisodeStep(state, action, reward));
                //         break;
                //     }
                // }
                episode.Add(new EpisodeStep(state, action, reward));
                state = new State(Level.Player.Position, boxesPos);
                action = policy[state];
                step++;
            }

            return episode;
        }

        public static double Average(List<double> list)
        {
            double sum = 0;
            foreach (double x in list)
                sum += x;
            return sum / list.Count;
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