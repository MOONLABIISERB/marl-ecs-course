namespace Classes
{
    public class State
    {
        public string Name { get; set; }
        public double Reward { get; set; }
        public State(string name, double reward)
        {
            Name = name;
            Reward = reward;
        }


        public override string ToString()
        {
            return Name;
        }
    }
}