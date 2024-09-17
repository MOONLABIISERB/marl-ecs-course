using Q2.Environment;
namespace Q2.RL
{
    public class StateAction
    {
        public State State { get; set; }
        public Action Action { get; set; }

        public StateAction(State state, Action action)
        {
            State = state;
            Action = action;
        }

        public override bool Equals(object obj)
        {
            if (obj is StateAction other)
            {
                return Equals(other);
            }
            return false;
        }

        public bool Equals(StateAction stateAction)
        {
            if (this.State == stateAction.State && this.Action == stateAction.Action)
                return true;
            return false;
        }

        public override int GetHashCode()
        {
            int hashCode = State.GetHashCode();
            hashCode = (hashCode * 222) ^ Action.GetHashCode();
            return hashCode;
        }

        public static bool operator == (StateAction s1, StateAction s2)
        {
            // Check if both are null or both refer to the same object
            if (ReferenceEquals(s1, s2))
            {
                return true;
            }

            // If one is null, but not both, return false
            if (s1 is null || s2 is null)
            {
                return false;
            }

            return s1.Equals(s2);
        }

        public static bool operator !=(StateAction s1, StateAction s2)
        {
            return !(s1 == s2);
        }
    }
}