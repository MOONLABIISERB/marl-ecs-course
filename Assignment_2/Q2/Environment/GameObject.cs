using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Q2.Extension;

namespace Q2.Environment
{
    public class GameObject
    {
        public Point Position { get; set; }
        public GameObjectType Type { get; set; }
        public Texture2D Sprite { get; set; }

        public GameObject(int x, int y, GameObjectType type, Texture2D sprite)
        {
            Position = new Point(x, y);
            Type = type;
            Sprite = sprite;
        }

        public void Move(int newX, int newY)
        {
            Position = new Point(newX, newY);
        }
    }
}