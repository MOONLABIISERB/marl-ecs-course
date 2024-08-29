namespace Classes
{
    public class Action
    {
        public string Name { get; set; }

        public Action(string name)
        {
            Name = name;
        }

        public override string ToString()
        {
            return Name;
        }
    }
}