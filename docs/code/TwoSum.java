import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class TwoSum {
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i, j};
                }
            }
        }
        return new int[0];
    }

    // A simple container to hold LeetCode-style test cases
    static class TestCase {
        int[] nums;
        int target;
        int[] expected;

        TestCase(int[] nums, int target, int[] expected) {
            this.nums = nums;
            this.target = target;
            this.expected = expected;
        }
    }

    public static void main(String[] args) {
        TwoSum sol = new TwoSum();

        // --- COPY & PASTE CASES HERE ---
        List<TestCase> cases = new ArrayList<>();
        
        // Example 1: nums = [2,7,11,15], target = 9, Output: [0,1]
        cases.add(new TestCase(new int[]{2, 7, 11, 15}, 9, new int[]{0, 1}));

        // Example 2: nums = [3,2,4], target = 6, Output: [1,2]
        cases.add(new TestCase(new int[]{3, 2, 4}, 6, new int[]{1, 2}));

        // Example 3: nums = [3,3], target = 6, Output: [0,1]
        cases.add(new TestCase(new int[]{3, 3}, 6, new int[]{0, 1}));
        // -------------------------------

        // Test Runner Logic
        System.out.println("Running LeetCode Test Cases...");
        for (int i = 0; i < cases.size(); i++) {
            TestCase tc = cases.get(i);
            int[] actual = sol.twoSum(tc.nums, tc.target);
            
            boolean passed = Arrays.equals(actual, tc.expected);
            
            System.out.printf("Test Case %d: %s\n", i + 1, passed ? "PASS ✅" : "FAIL ❌");
            if (!passed) {
                System.out.println("   Input: nums = " + Arrays.toString(tc.nums) + ", target = " + tc.target);
                System.out.println("   Expected: " + Arrays.toString(tc.expected));
                System.out.println("   Actual:   " + Arrays.toString(actual));
            }
        }
    }
}