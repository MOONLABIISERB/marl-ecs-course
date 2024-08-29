
namespace Classes
{
    public class Transition
    {
        public State NextState { get; set; }
        public double Probability { get; set; }
        public double Reward { get; set; }

        public Transition(State nextState, double probability)
        {
            NextState = nextState;
            Probability = probability;
            Reward = nextState.Reward;
        }
    }
}