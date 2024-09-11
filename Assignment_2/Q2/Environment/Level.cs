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
        public GameObject[] Goals{ get; set; }
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
            Goals = new GameObject[coordinates.Length];
            for (int i = 0; i < coordinates.Length; i++)
            {
                Goals[i] = new GameObject(coordinates[i].X, coordinates[i].Y, GameObjectType.Goal, GoalTexture);
                Map[coordinates[i].X, coordinates[i].Y] = Goals[i];
            }
        }

        public void AddWalls(params Point[] coordinates)
        {
            for (int i = 0; i < coordinates.Length; i++)
            {
                Map[coordinates[i].X, coordinates[i].Y] = new GameObject(coordinates[i].X, coordinates[i].Y, GameObjectType.Wall, WallTexture);
            }
        }

        public void InitializeLevel(Point[] walls, Point[] goals, Point[] boxes, Point player)
        {
            AddWalls(walls);
            AddGoals(goals);
            AddBoxes(boxes);
            AddPlayer(player.X, player.Y);
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

        public void MovePlayer(Action action)
        {
            int deltaX = 0;
            int deltaY = 0;
            switch (action)
            {
                case Action.Up:
                    deltaY = 1;
                    break;
                case Action.Down:
                    deltaY = -1;
                    break;
                case Action.Left:
                    deltaX = -1;
                    break;
                case Action.Right:
                    deltaX = 1;
                    break;
            }



            Point newPlayerPos = new Point(Player.Position.X + deltaX, Player.Position.Y + deltaY);

            if (CheckOutOfBound(newPlayerPos))
                return;            
                
            // Move Player, if box is there move box or move player again
            if (Map[newPlayerPos.X, newPlayerPos.Y].Type != GameObjectType.Wall)
            {
                Player.Position = newPlayerPos;
            }

            //Check if newPlayerPos is box
            foreach (var box in Boxes)
            {
                if (newPlayerPos == box.Position)
                {
                    Point newBoxPos = new Point(newPlayerPos.X + deltaX, newPlayerPos.Y + deltaY);

                    if (!CheckOutOfBound(newBoxPos) &&
                         Map[newBoxPos.X, newBoxPos.Y].Type != GameObjectType.Wall &&
                         !CheckBoxOverlapping(newBoxPos))
                    {
                        Player.Position = newPlayerPos;
                        box.Position = newBoxPos;
                    }
                    else
                    {
                        Player.Position = new Point(newPlayerPos.X - deltaX, newPlayerPos.Y - deltaY);
                    }
                }
            }


        }

        bool CheckOutOfBound(Point pos)
        {
            if (!(pos.X >= 0 && pos.X < Columns && pos.Y >= 0 && pos.Y < Rows))
            {
                return true;
            }
            return false;
        }

        bool CheckBoxOverlapping(Point pos)
        {
            foreach (var box in Boxes)
            {
                if (box.Position == pos)
                    return true;
            }
            return false;
        }

        public bool LevelComplete()
        {
            for (int i = 0; i < Boxes.Length; i++)
            {
                if (!(Boxes[i].Position == Goals[i].Position))
                {
                    return false;
                }
            }
            return true;
        }

        // public bool ReachedGoal(GameObject box)
        // {
        //     if (Map[box.Position.X, box.Position.Y].Type == GameObjectType.Goal)
        //         return true ;
        //     return false;
        // }

        public static bool UnMovable(GameObject box)
        {
            if (box.Position != new Point(1,5) ||
                box.Position != new Point(2,5) ||
                box.Position != new Point(1,1) ||
                box.Position != new Point(2,1) ||
                box.Position != new Point(4,2) ||
                box.Position != new Point(4,3))
                return true;
            return false;
        } 
    }
}