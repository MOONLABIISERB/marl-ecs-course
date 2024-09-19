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

        public static T[] Shuffle<T>(T[] array)
        {
            Random rng = new Random();
            int n = array.Length;

            // Create a copy of the array so we don't modify the original array
            T[] shuffledArray = (T[])array.Clone();

            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = shuffledArray[k];
                shuffledArray[k] = shuffledArray[n];
                shuffledArray[n] = value;
            }

            // Return the shuffled array
            return shuffledArray;
        }


    }
}