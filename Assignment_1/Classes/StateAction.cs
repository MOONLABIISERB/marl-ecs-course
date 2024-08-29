namespace Classes
{
    public class StateAction
    {
        public State CurrentState { get; set; }
        public Action ActionTaken { get; set; }
        public List<Transition> Transitions { get; set; }

        public StateAction(State currentState, Action actionTaken)
        {
            CurrentState = currentState;
            ActionTaken = actionTaken;
            Transitions = new List<Transition>();
        }

        public void AddTransition(State nextState, double probability)
        {
            Transitions.Add(new Transition(nextState, probability));
        }
    }
}