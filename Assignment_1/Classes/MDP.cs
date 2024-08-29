namespace Classes
{
    public class MDP
    {
        public List<StateAction> StateActions { get; set; }
        public Dictionary<State, double> OptimalValue;
        public Dictionary<State, Action> OptimalAction;

        private StateSpace StateSpace;
        private ActionSpace ActionSpace;

        public List<State> States;
        public List<Action> Actions;
        public MDP(StateSpace stateSpace, ActionSpace actionSpace)
        {
            StateActions = new List<StateAction>();

            OptimalValue = new Dictionary<State,double>();
            OptimalAction = new Dictionary<State,Action>();

            StateSpace = stateSpace;
            ActionSpace = actionSpace;

            States = StateSpace.GetStates();
            Actions = ActionSpace.GetActions();
        }

        public void AddStateAction(State currentState, Action actionTaken, List<Transition> transitions)
        {
            var stateAction = new StateAction(currentState, actionTaken);
            stateAction.Transitions.AddRange(transitions);
            StateActions.Add(stateAction);
        }


        public List<StateAction> GetStateActions(State state)
        {
            return StateActions.Where(sa => sa.CurrentState == state).ToList();
        }

        public void ValueIteration(out Dictionary<State, double> value, out Dictionary<State, Action> policy, double gamma, double epsilon)
        {
            bool isConverged = false;
            // Initialize Policies and values
            InitializePolicyAndValue(out policy,out value);

            while (!isConverged)
            {
                isConverged = true;
                Dictionary<State, double> VNew = new Dictionary<State, double>();

                foreach (var state in States)
                {
                    double maxValue = double.NegativeInfinity;
                    Action bestAction = Actions[0];

                    foreach (var stateAction in GetStateActions(state))
                    {
                        double actionValue = state.Reward;

                        foreach (var transition in stateAction.Transitions)
                        {
                            actionValue += transition.Probability *  gamma * value[transition.NextState];
                        }

                        if (actionValue > maxValue)
                        {
                            maxValue = actionValue;
                            bestAction = stateAction.ActionTaken;
                        }
                    }

                    VNew[state] = maxValue;
                    policy[state] = bestAction;

                    if (Math.Abs(VNew[state] - value[state]) > epsilon)
                    {
                        isConverged = false;
                    }
                }
                value = VNew;
            }
        }
        public void PolicyIteration(out Dictionary<State, double> value, out Dictionary<State, Action> policy, double gamma, double epsilon)
        {
            InitializePolicyAndValue(out policy, out value);

            bool policyStable = false;

            while (!policyStable)
            {
                PolicyEvaluation(ref value, policy, gamma, epsilon);

                policyStable = true;
                PolicyImprovement(ref value, ref policy, ref policyStable, gamma);
            }
        }
        void PolicyEvaluation(ref Dictionary<State, double> value, Dictionary<State, Action> policy, double gamma, double epsilon)
        {
            bool isConverged = false;

            while (!isConverged)
            {
                isConverged = true;
                Dictionary<State, double> VNew = new Dictionary<State, double>();

                foreach (var state in States)
                {
                    double newValue = state.Reward;

                    var stateAction = GetStateActions(state).FirstOrDefault(sa => sa.ActionTaken == policy[state]);

                    if (stateAction != null)
                    {
                        foreach (var transition in stateAction.Transitions)
                        {
                            newValue += transition.Probability * (gamma * value[transition.NextState]);
                        }
                    }

                    VNew[state] = newValue;

                    if (Math.Abs(VNew[state] - value[state]) > epsilon)
                    {
                        isConverged = false;
                    }
                }

                value = VNew;
            }

        }

        void PolicyImprovement(ref Dictionary<State, double> value, ref Dictionary<State, Action> policy, ref bool policyStable, double gamma)
        {

            foreach (var state in States)
            {
                double maxValue = double.NegativeInfinity;
                Action bestAction = policy[state];  // Start with the current policy action

                foreach (var stateAction in GetStateActions(state))
                {
                    double actionValue = 0.0;

                    foreach (var transition in stateAction.Transitions)
                    {
                        actionValue += transition.Probability * (transition.Reward + gamma * value[transition.NextState]);
                    }

                    if (actionValue > maxValue)
                    {
                        maxValue = actionValue;
                        bestAction = stateAction.ActionTaken;
                    }
                }

                // If the best action is different from the current action, the policy is not stable
                if (policy[state] != bestAction)
                {
                    policy[state] = bestAction;
                    policyStable = false;
                }
            }
        }
    
        private void InitializePolicyAndValue(out Dictionary<State, Action> policy, out Dictionary<State, double> value)
        {
            // Initialize Policies and values
            policy = new Dictionary<State, Action>();
            value = new Dictionary<State, double>();
            
            foreach (var state in States)
            {
                policy[state] = Actions[0];
                value[state] = 0;
            }
        }
    }
}