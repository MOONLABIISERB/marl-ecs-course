import java.util.*;

public class Main {
    private static int[][] movability = new int[9][9];

    private static int[] addAction(List<Integer> state, int action){
        if (action == 0){
            return new int[]{state.get(0) - 1, state.get(1)}; // Up
        }
        else if (action == 1){
            return new int[]{state.get(0) + 1, state.get(1)}; // Down
        }
        else if (action == 2){
            return new int[]{state.get(0), state.get(1) + 1}; // Right
        }
        else if (action == 3){
            return new int[]{state.get(0), state.get(1) - 1}; // Left
        }
        return new int[0];
    }

    private static boolean accessible(int[] state){
        if (state[0] < 0 || state[0] >= 9 || state[1] < 0 || state[1] >= 9) {
            return false;
        }
        return movability[state[0]][state[1]] == 1;
    }

    private static List<Integer> getNewState(List<Integer> state, int action ){
        // Portal entry
        if (state.get(0) == 2 && state.get(1) == 2){
            return Arrays.asList(6, 6); // Portal exit
        }
        int[] newState = addAction(state, action);
        if (accessible(newState)){
            return Arrays.asList(newState[0], newState[1]);
        } else {
            return state;
        }
    }

    private static double getReward(List<Integer> state){
        return (state.get(0) == 8 && state.get(1) == 8) ? 1.0 : 0.0;
    }

    public static void main(String[] args){
        // action integer to word map
        HashMap<Integer, String> actionMap = new HashMap<>();
        actionMap.put(0, "Up");
        actionMap.put(1, "Down");
        actionMap.put(2, "Right");
        actionMap.put(3, "Left");

        //actions
        int[] actions = {0, 1, 2, 3};

        // List of all possible states
        ArrayList<List<Integer>> states = new ArrayList<>();
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                states.add(Arrays.asList(i, j));
            }
        }

        //initialize movability array
        for (int i = 0; i < 9; i++){
            Arrays.fill(movability[i], 1);
        }

        ArrayList<int[]> walls = new ArrayList<>();
        Collections.addAll(walls,
                new int[]{0, 3}, new int[]{3, 0},
                new int[]{1,3}, new int[]{2,3}, new int[]{3,3},
                new int[]{3,1}, new int[]{5,8}, new int[]{5,7},
                new int[]{5,6}, new int[]{5,5}, new int[]{6,5},
                new int[]{7,5}, new int[]{8,5}, new int[]{3,2}
        );

        for(int[] wall : walls){
            movability[wall[0]][wall[1]] = 0;
        }

        //value map
        HashMap<List<Integer>, Double> values = new HashMap<>();
        for(List<Integer> state : states){
            values.put(state, 0.0);
        }
        values.put(Arrays.asList(8, 8), 1.0); // Terminal state

        double gamma = 0.9;
        double theta = 0.01;
        double delta;

        do {
            delta = 0;
            for (List<Integer> state : states){
                if (state.get(0) == 8 && state.get(1) == 8) continue; // Skip terminal state
                double v = values.get(state);
                double maxValue = Double.NEGATIVE_INFINITY;

                for(int action : actions){
                    List<Integer> newState = getNewState(state, action);
                    double value = getReward(newState) + gamma * values.get(newState);
                    maxValue = Math.max(maxValue, value);
                }

                values.put(state, maxValue);
                delta = Math.max(delta, Math.abs(v - maxValue));
            }
        } while (delta > theta);

        // Print the final values
        System.out.println("Final state values:");
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                System.out.printf("%6.3f ", values.get(Arrays.asList(i, j)));
            }
            System.out.println();
        }

        // Print the optimal policy
        System.out.println("\nOptimal Policy:");
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if (i == 8 && j == 8) {
                    System.out.print("  G  "); // Goal
                } else if (i == 2 && j == 2) {
                    System.out.print("  I  "); // Portal In
                } else if (i == 6 && j == 6) {
                    System.out.print("  O  "); // Portal Out
                } else if (movability[i][j] == 0) {
                    System.out.print("  #  "); // Wall
                } else {
                    double maxValue = Double.NEGATIVE_INFINITY;
                    int bestAction = -1;
                    for (int action : actions) {
                        List<Integer> newState = getNewState(Arrays.asList(i, j), action);
                        double value = getReward(newState) + gamma * values.get(newState);
                        if (value > maxValue) {
                            maxValue = value;
                            bestAction = action;
                        }
                    }
                    System.out.print("  " + actionMap.get(bestAction).charAt(0) + "  ");
                }
            }
            System.out.println();
        }
    }
}