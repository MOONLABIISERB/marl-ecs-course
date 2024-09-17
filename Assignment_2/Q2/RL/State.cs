using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;

namespace Q2.RL
{
    public class State
    {
        public Point PlayerPosition { get; set; }
        public Point[] BoxesPosition { get; set; }

        public State(Point[] pos)
        {
            PlayerPosition = pos[0];
            BoxesPosition = new Point[pos.Length - 1];
            for (int i = 1; i < pos.Length; i++)
            {
                BoxesPosition[i-1] = pos[i];
            }
        }

        public State(Point playerPos, Point[] boxPos)
        {
            PlayerPosition = playerPos;
            BoxesPosition = new Point[boxPos.Length];
            for (int i = 0; i < boxPos.Length; i++)
            {
                BoxesPosition[i] = boxPos[i];
            }
        }

        public override bool Equals(object obj)
        {
            if (obj is State other)
            {
                return Equals(other);
            }
            return false;
        }
        public bool Equals(State state)
        {
            if (state.PlayerPosition != this.PlayerPosition)
                return false;
            if (BoxesPosition.Length != state.BoxesPosition.Length)
                return false;
            for (int i = 0; i < BoxesPosition.Length; i++)
            {
                if (this.BoxesPosition[i] != state.BoxesPosition[i])
                    return false;
            }
            return true;
        }

        public override int GetHashCode()
        {
            int hashCode = PlayerPosition.GetHashCode();
            foreach (var point in BoxesPosition)
            {
                hashCode = (hashCode * 397) ^ point.GetHashCode();
            }
            return hashCode;
        }

        public static bool operator == (State s1, State s2)
        {
            // Check if both are null or both refer to the same object
            if (ReferenceEquals(s1, s2))
            {
                return true;
            }

            // If one is null, but not both, return false
            if (s1 is null || s2 is null)
            {
                return false;
            }

            return s1.Equals(s2);
        }

        public static bool operator !=(State s1, State s2)
        {
            return !(s1 == s2);
        }
    }
}