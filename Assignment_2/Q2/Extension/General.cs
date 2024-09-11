using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace Q2.Extension
{
    public static class General
    {
        public static Vector2 GetPixelCoord(Point point, int TileSize, int Rows)
        {
            return new Vector2(point.X * TileSize, (Rows - 1 - point.Y) * TileSize);
        }

        public static bool IsKeyJustPressed(Keys key, KeyboardState currentKeyboardState, KeyboardState previousKeyboardState)
        {
            // Return true if the key is currently down but was not down in the previous frame
            return currentKeyboardState.IsKeyDown(key) && previousKeyboardState.IsKeyUp(key);
        }

    }
}