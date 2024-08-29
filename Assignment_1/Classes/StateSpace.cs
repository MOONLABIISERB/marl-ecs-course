using System.Dynamic;

namespace Classes
{
    public class StateSpace: DynamicObject
    {
        private Dictionary<string, State> states;

        public StateSpace()
        {
            states = new Dictionary<string, State>();
        }
        // Shorthand method to allow 's1.Add("Sleep")'
        public void Add(string stateName, double stateReward)
        {
            states.Add(stateName, new State(stateName, stateReward));
        }

        public void Add(State state)
        {
            states.Add(state.Name, state);
        }

        // Overriding TryGetMember to access states dynamically
        public override bool TryGetMember(GetMemberBinder binder, out object? result)
        {
            if (states.TryGetValue(binder.Name, out State? state))
            {
                result = state;
                return true;
            }

            result = null;
            return false;
        }

        public List<State> GetStates()
        {
            return states.Values.ToList();
        }
    }
}