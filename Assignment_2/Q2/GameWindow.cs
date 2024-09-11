using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Q2.Environment;
using Q2.Extension;

namespace Q2;


// TODO: State Enter - (which state from, player or box), State Exit - (which state to, player or box)
public class GameWindow : Game
{
    int TileSize = 64;
    int GridRows = 7;
    int GridColumns = 6;
    private GraphicsDeviceManager _graphics;
    private SpriteBatch _spriteBatch;
    public Level level;

    KeyboardState currentKeyboardState;
    KeyboardState previousKeyboardState;

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
    }

    protected override void Update(GameTime gameTime)
    {
        currentKeyboardState = Keyboard.GetState();

        if (General.IsKeyJustPressed(Keys.W, currentKeyboardState, previousKeyboardState) ||
            General.IsKeyJustPressed(Keys.Up, currentKeyboardState, previousKeyboardState))
        {
            MovePlayer(Action.Up);
        }
        else if (General.IsKeyJustPressed(Keys.A, currentKeyboardState, previousKeyboardState) ||
                 General.IsKeyJustPressed(Keys.Left, currentKeyboardState, previousKeyboardState))
        {
            MovePlayer(Action.Left);
        }
        else if (General.IsKeyJustPressed(Keys.S, currentKeyboardState, previousKeyboardState) || 
                 General.IsKeyJustPressed(Keys.Down, currentKeyboardState, previousKeyboardState))
        {
            MovePlayer(Action.Down);
        }
        else if (General.IsKeyJustPressed(Keys.D, currentKeyboardState, previousKeyboardState) ||
                 General.IsKeyJustPressed(Keys.Right, currentKeyboardState, previousKeyboardState))
        {
            MovePlayer(Action.Right);
        }

        previousKeyboardState = currentKeyboardState;

        if (LevelComplete())
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

    void InitializeLevel()
    {

        Point[] walls = {new Point(0,0),
                              new Point(0,1),
                              new Point(0,2),
                              new Point(0,3),
                              new Point(0,4),
                              new Point(0,5),
                              new Point(0,6),
                              new Point(1,0),
                              new Point(1,6),
                              new Point(2,0),
                              new Point(2,6),
                              new Point(3,0),
                              new Point(3,1),
                              new Point(3,4),
                              new Point(3,5),
                              new Point(3,6),
                              new Point(4,1),
                              new Point(4,4),
                              new Point(5,1),
                              new Point(5,2),
                              new Point(5,3),
                              new Point(5,4)};
        level.AddWalls(walls);

        level.AddGoals(new Point(1,3));
        level.AddBoxes(new Point(3,2));
        level.AddPlayer(2, 5);
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

        Point newPlayerPos = new Point(level.Player.Position.X + deltaX, level.Player.Position.Y + deltaY);
        // Move Player, if box is there move box or move player again
        if (level.Map[newPlayerPos.X, newPlayerPos.Y].Type != GameObjectType.Wall)
        {
            level.Player.Position = newPlayerPos;
        }

        //Check if newPlayerPos is box
        foreach (var box in level.Boxes)
        {
            if (newPlayerPos == box.Position)
            {
                Point newBoxPos = new Point(newPlayerPos.X + deltaX, newPlayerPos.Y + deltaY);
                if (level.Map[newBoxPos.X, newBoxPos.Y].Type != GameObjectType.Wall)
                {
                    level.Player.Position = newPlayerPos;
                    box.Position = newBoxPos;
                }
                else
                {
                    level.Player.Position = new Point(newPlayerPos.X - deltaX, newPlayerPos.Y - deltaY);
                }
            }
        }


    }

    public bool LevelComplete()
    {
        foreach (var box in level.Boxes)
        {
            if (level.Map[box.Position.X, box.Position.Y].Type != GameObjectType.Goal &&
                box.Position != new Point(1,5) &&
                box.Position != new Point(2,5) &&
                box.Position != new Point(1,1) &&
                box.Position != new Point(2,1) &&
                box.Position != new Point(4,2) &&
                box.Position != new Point(4,3))
            {
                return false;
            }
        }
        return true;
    }
    
}
