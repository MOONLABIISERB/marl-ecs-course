using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security.Authentication.ExtendedProtection;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Q2.Extension;

namespace Q2.Environment
{
    public class Level
    {
        public int TileSize { get; set; }
        public GameObject[,] Map { get; set; }
        public GameObject Player { get; set; }        
        public GameObject[] Boxes { get; set; }
        Texture2D PlayerTexture, BoxTexture, FloorTexture, WallTexture, GoalTexture;
        public int Rows => Map.GetLength(1);
        public int Columns => Map.GetLength(0);

        public Level(int rows, int columns, int tileSize, Texture2D playerTexture, Texture2D boxTexture, Texture2D floorTexture, Texture2D wallTexture, Texture2D goalTexture)
        {
            TileSize = tileSize;
            Map = new GameObject[columns, rows];
            PlayerTexture = playerTexture;
            BoxTexture = boxTexture;
            FloorTexture = floorTexture;
            WallTexture = wallTexture;
            GoalTexture = goalTexture;
            for (int i = 0; i < columns; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    Map[i,j] = new GameObject(i, j, GameObjectType.Floor, FloorTexture);
                }
            }
            
        }

        public void AddPlayer(int x, int y)
        {
            Player = new GameObject(x, y, GameObjectType.Player, PlayerTexture);
        }

        public void AddBoxes(params Point[] coordinates)
        {
            Boxes = new GameObject[coordinates.Length];
            for (int i = 0; i < Boxes.Length; i++)
            {
                Boxes[i] = new GameObject(coordinates[i].X, coordinates[i].Y, GameObjectType.Box, BoxTexture);
            }
        }

        public void AddGoals(params Point[] coordinates)
        {
            for (int i = 0; i < coordinates.Length; i++)
            {
               Map[coordinates[i].X, coordinates[i].Y] = new GameObject(coordinates[i].X, coordinates[i].Y, GameObjectType.Goal, GoalTexture);
            }
        }

        public void AddWalls(params Point[] coordinates)
        {
            for (int i = 0; i < coordinates.Length; i++)
            {
                Map[coordinates[i].X, coordinates[i].Y] = new GameObject(coordinates[i].X, coordinates[i].Y, GameObjectType.Wall, WallTexture);
            }
        }

        public void DrawLevel(SpriteBatch spriteBatch)
        {
            for (int row = 0; row < Rows; row++)
            {
                for (int column = 0; column < Columns; column++)
                {
                    Vector2 pos = General.GetPixelCoord(Map[column, row].Position, TileSize, Rows);
                    spriteBatch.Draw(FloorTexture, pos, Color.White);
                    DrawGameObject(Map[column, row], spriteBatch, FloorTexture);
                }
            }

            // Draw Player
            DrawGameObject(Player, spriteBatch, FloorTexture);

            // Draw Boxes
            foreach (var box in Boxes)
            {
                DrawGameObject(box, spriteBatch, FloorTexture);    
            }
        }

        void DrawGameObject(GameObject gameObject, SpriteBatch spriteBatch, Texture2D floorTexture)
        {
            
            Vector2 gameObjectPos = General.GetPixelCoord(gameObject.Position, TileSize, Rows);
            gameObjectPos.X += (floorTexture.Width - gameObject.Sprite.Width) / 2f;
            gameObjectPos.Y += (floorTexture.Height - gameObject.Sprite.Height) / 2f;
            spriteBatch.Draw(gameObject.Sprite, gameObjectPos, Color.White);
        }
    }
}