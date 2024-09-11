using System.Collections.Generic;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Q2.Environment;
using Q2.Extension;
using Q2.RL;

namespace Q2;


// TODO: State Enter - (which state from, player or box), State Exit - (which state to, player or box)
public class GameWindow : Game
{
    bool RLEnabled = true;
    MDP mdp;
    int TileSize = 64;
    int GridRows = 7;
    int GridColumns = 6;
    private GraphicsDeviceManager _graphics;
    private SpriteBatch _spriteBatch;
    public Level level;

    int currentFrame = 0;
    int framesToWait = 20;
    KeyboardState currentKeyboardState;
    KeyboardState previousKeyboardState;

    Dictionary<State, double> OptimalValue = new Dictionary<State, double>();
    Dictionary<State, Action> OptimalPolicy = new Dictionary<State, Action>();

    public GameWindow()
    {
        _graphics = new GraphicsDeviceManager(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

    }

    protected override void Initialize()
    {

        _graphics.PreferredBackBufferWidth = GridColumns * TileSize;
        _graphics.PreferredBackBufferHeight = GridRows * TileSize;
        _graphics.ApplyChanges();

        base.Initialize();
    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);

        level = new Level(GridRows, GridColumns, TileSize,
                          Content.Load<Texture2D>("Character"),
                          Content.Load<Texture2D>("CrateDark_Brown"),
                          Content.Load<Texture2D>("Ground_Sand"),
                          Content.Load<Texture2D>("WallRound_Brown"),
                          Content.Load<Texture2D>("EndPoint_Blue"));

        InitializeLevel();

        if (RLEnabled)
        {
            mdp = new MDP(this,level);
            mdp.ValueIteration(out OptimalValue, out OptimalPolicy, 1, (1 + level.Boxes.Length) * 25 + 10);

            InitializeLevel();
        }
    }

    protected override void Update(GameTime gameTime)
    {
        if (!RLEnabled)
        {
            currentKeyboardState = Keyboard.GetState();

            if (General.IsKeyJustPressed(Keys.W, currentKeyboardState, previousKeyboardState) ||
                General.IsKeyJustPressed(Keys.Up, currentKeyboardState, previousKeyboardState))
            {
                level.MovePlayer(Action.Up);
            }
            else if (General.IsKeyJustPressed(Keys.A, currentKeyboardState, previousKeyboardState) ||
                    General.IsKeyJustPressed(Keys.Left, currentKeyboardState, previousKeyboardState))
            {
                level.MovePlayer(Action.Left);
            }
            else if (General.IsKeyJustPressed(Keys.S, currentKeyboardState, previousKeyboardState) || 
                    General.IsKeyJustPressed(Keys.Down, currentKeyboardState, previousKeyboardState))
            {
                level.MovePlayer(Action.Down);
            }
            else if (General.IsKeyJustPressed(Keys.D, currentKeyboardState, previousKeyboardState) ||
                    General.IsKeyJustPressed(Keys.Right, currentKeyboardState, previousKeyboardState))
            {
                level.MovePlayer(Action.Right);
            }

            previousKeyboardState = currentKeyboardState;
        }
        else
        {
            if (currentFrame < framesToWait)
            {
                currentFrame++;
            }
            else
            {
                if (!level.LevelComplete())
                {
                    List<Point> points = new List<Point>();
                    points.Add(level.Player.Position);
                    foreach(var box in level.Boxes)
                    {
                        points.Add(box.Position);
                    }
                    State key = new State(points.ToArray());
                    System.Console.WriteLine($"Value at X:{level.Player.Position.X} Y:{level.Player.Position.Y} = {OptimalValue[key]}");
                    level.MovePlayer(OptimalPolicy[key]);
                }
                currentFrame = 0;
            }
            
        }

        if (level.LevelComplete())
        {
            InitializeLevel();
        }
            
        base.Update(gameTime);
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.CornflowerBlue);

        _spriteBatch.Begin();

        level.DrawLevel(_spriteBatch);
        
        _spriteBatch.End();
        base.Draw(gameTime);
    }

    public void InitializeLevel()
    {
        
        // Level 1

        // Point[] walls = {new Point(0,0),
        //                  new Point(0,1),
        //                  new Point(0,2),
        //                  new Point(0,3),
        //                  new Point(0,4),
        //                  new Point(0,5),
        //                  new Point(0,6),
        //                  new Point(1,0),
        //                  new Point(1,6),
        //                  new Point(2,0),
        //                  new Point(2,6),
        //                  new Point(3,0),
        //                  new Point(3,1),
        //                  new Point(3,4),
        //                  new Point(3,5),
        //                  new Point(3,6),
        //                  new Point(4,1),
        //                  new Point(4,4),
        //                  new Point(5,1),
        //                  new Point(5,2),
        //                  new Point(5,3),
        //                  new Point(5,4)};
        
        // Point[] goals = {new Point(1,3)};
        
        // Point[] boxes = {new Point(3,2)};
        
        // Point player = new Point(2,5);

        


        // Level 2

        // Point[] walls = {new Point(2,2),
        //                  new Point(2,4)};

        // Point[] goals = {new Point(1,2),
        //                  new Point(1,4)};

        // Point[] boxes = {new Point(3,2),
        //                  new Point(3,4)};

        // Point player = new Point(5, 3);


        // Level 3
        // Point[] walls = {new Point(1,2),
        //                  new Point(4,4)};

        // Point[] goals = {new Point(1,0),
        //             new Point(4,5)};

        // Point[] boxes = {new Point(4,0),
        //                  new Point(4,1)};

        // Point player = new Point(2, 2);

        // Level 4

        // Point[] walls = {new Point(0,3),
        //                  new Point(0,4),
        //                  new Point(0,5),
        //                  new Point(0,6),
        //                  new Point(1,2),
        //                  new Point(1,3),
        //                  new Point(1,6),
        //                  new Point(2,6),
        //                  new Point(3,2),
        //                  new Point(3,3),
        //                  new Point(3,5),
        //                  new Point(3,6),
        //                  new Point(4,1),
        //                  new Point(4,5),
        //                  new Point(5,3),
        //                  new Point(5,4),
        //                  new Point(5,5)};

        // Point[] goals = {new Point(0,2),
        //                  new Point(0,1),
        //                  new Point(0,0)};

        // Point[] boxes = {new Point(1,1),
        //                  new Point(2,4),
        //                  new Point(4,0)};

        // Point player = new Point(1,5);

        // Level 5

        Point[] walls = {new Point(0,0),
                         new Point(0,4),
                         new Point(0,5),
                         new Point(0,6),
                         new Point(1,0),
                         new Point(1,6),
                         new Point(2,0),
                         new Point(2,2),
                         new Point(2,4),
                         new Point(2,6),
                         new Point(3,0),
                         new Point(3,4),
                         new Point(3,6),
                         new Point(4,0),
                         new Point(4,1),
                         new Point(4,6),
                         new Point(5,1),
                         new Point(5,5),
                         new Point(5,6)};

        Point[] goals = {new Point(1,3),
                         new Point(1,4),
                         new Point(2,3)};

        Point[] boxes = {new Point(4,4),
                         new Point(3,3),
                         new Point(3,2)};

        Point player = new Point(2,1);

        // Create Level
        level.AddWalls(walls);
        level.AddGoals(goals);
        level.AddBoxes(boxes);
        level.AddPlayer(player.X, player.Y);
    }
    
}
