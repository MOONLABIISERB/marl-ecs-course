import java.util.HashMap;

public class Main {
    public static void main(String[] args){
        //state map
        HashMap<Integer, String> stateMap = new HashMap<>();
        stateMap.put(0, "Academic Block");
        stateMap.put(1, "Canteen");
        stateMap.put(2, "Hostel");

        HashMap<Integer, String> actionMap = new HashMap<>();
        actionMap.put(0, "Hungry");
        actionMap.put(1, "Full");

        //initialise v values
        double[] value = new double[3]; //3 states -> ab, canteen, hostel
        value[0] = 0;
        value[1] = 0;
        value[2] = 0;

        int[] reward = new int[3]; // initialise rewards
        reward[0] = 3;
        reward[1] = 1;
        reward[2] = -1;

        double gamma = 0.9;


        //[ab, canteen, hostel][hungry, full][ab, canteen, hostel];
        double[][][] probs = new double[3][2][3];
        //set non zero probabilities
        probs[2][0][1] = 1;
        probs[2][1][0] = 0.5;
        probs[2][1][2] = 0.5;
        probs[0][1][0] = 0.7;
        probs[0][1][1] = 0.3;
        probs[0][0][1] = 0.8;
        probs[0][0][0] = 0.2;
        probs[1][1][0] = 0.6;
        probs[1][1][2] = 0.3;
        probs[1][1][1] = 0.1;
        probs[1][0][1] = 1;

//        // Set probabilities based on the problem statement
//        probs[0][1][0] = 0.7; probs[0][1][1] = 0.3; // Academic Block, Full
//        probs[0][0][1] = 0.8; probs[0][0][0] = 0.2; // Academic Block, Hungry
//        probs[1][1][0] = 0.6; probs[1][1][2] = 0.3; probs[1][1][1] = 0.1; // Canteen, Full
//        probs[1][0][1] = 1; // Canteen, Hungry
//        probs[2][1][0] = 0.5; probs[2][1][2] = 0.5; // Hostel, Full
//        probs[2][0][1] = 1; // Hostel, Hungry

        double theta = 0.00001;

        double delta = Double.POSITIVE_INFINITY;
        double v;
        double val;
        double[] candidates = new double[2];

        //value iteration
        while (delta > theta){
            delta = 0;
            for(int state = 0; state < 3; state++){
                v = value[state];

                for(int action = 0; action < 2; action++){
                    val = 0;
                    for(int state_ = 0; state_ < 3; state_++){
                        val += probs[state][action][state_] * (reward[state_] + gamma*value[state_]);
                    }
                    candidates[action] = val;
                }
                value[state] = Math.max(candidates[0], candidates[1]);
                delta = Math.max(delta, Math.abs(v - value[state]));


            }
        }

        int[] policy = {0, 0, 0};
        //get optimal policy
        for(int state = 0; state < 3; state++){
            double max = Double.NEGATIVE_INFINITY;

            for(int action = 0; action < 2; action++) {
                val = 0;
                for (int state_ = 0; state_ < 3; state_++) {
                    val += probs[state][action][state_] * (reward[state_] + gamma * value[state_]);
                }
                if (val > max) {
                    policy[state] = action;
                }
            }

        }
        System.out.println("Optimal policy");
        for(int state = 0; state < 3; state++){
            System.out.println("Optimal action for "+stateMap.get(state)+" : "+actionMap.get(policy[state]));
        }

        System.out.println("\n\n Values");
        for(int state = 0; state < 3; state++){
            System.out.println("Value for "+stateMap.get(state)+" : "+value[state]);
        }

    }
}
