using Q2.Environment;

namespace Q2.RL
{
    public class EpisodeStep
    {
        public State State { get; set; }
        public Action Action { get; set; }
        public double Reward { get; set; }
        public EpisodeStep(State state, Action action, double reward)
        {
            State = state;
            Action = action;
            Reward = reward;
        }
    }
}