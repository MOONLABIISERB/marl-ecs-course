using System.Dynamic;

namespace Classes
{
    public class ActionSpace : DynamicObject
    {
        private Dictionary<string, Action> actions;

        public ActionSpace()
        {
            actions = new Dictionary<string, Action>();
        }
        // Shorthand method to allow 's1.Add("Sleep")'
        public void Add(string actionName)
        {
            actions.Add(actionName, new Action(actionName));
        }

        // Overriding TryGetMember to access states dynamically
        public override bool TryGetMember(GetMemberBinder binder, out object? result)
        {
            if (actions.TryGetValue(binder.Name, out Action? action))
            {
                result = action;
                return true;
            }

            result = null;
            return false;
        }

        public List<Action> GetActions()
        {
            return actions.Values.ToList();
        }
    }
}