error id: file://<WORKSPACE>/docs/code/LongestSubstringWithoutRepeatingCharacters.java:_empty_/LongestSubstringWithoutRepeatingCharacters#longestSubstringWithoutRepeatingCharacters#
file://<WORKSPACE>/docs/code/LongestSubstringWithoutRepeatingCharacters.java
empty definition using pc, found symbol in pc: _empty_/LongestSubstringWithoutRepeatingCharacters#longestSubstringWithoutRepeatingCharacters#
empty definition using semanticdb
empty definition using fallback
non-local guesses:

offset: 1527
uri: file://<WORKSPACE>/docs/code/LongestSubstringWithoutRepeatingCharacters.java
text:
```scala
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class LongestSubstringWithoutRepeatingCharacters {
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

    // 2. THE TEST CASE CONTAINER (Modified for String input and int output)
    static class TestCase {
        String s;
        int expected;

        TestCase(String s, int expected) {
            this.s = s;
            this.expected = expected;
        }
    }

    public static void main(String[] args) {
        LongestSubstringWithoutRepeatingCharacters sol = new LongestSubstringWithoutRepeatingCharacters();

        // 3. THE TEST DATA (Directly from LeetCode Markdown)
        List<TestCase> cases = new ArrayList<>();
        
        // Input: s = "abcabcbb", Output: 3
        cases.add(new TestCase("abcabcbb", 3));

        // Input: s = "bbbbb", Output: 1
        cases.add(new TestCase("bbbbb", 1));

        // Input: s = "pwwkew", Output: 3
        cases.add(new TestCase("pwwkew", 3));

        // 4. THE RUNNER (Simplified comparison for integers)
        System.out.println("Running LeetCode Test Cases...");
        for (int i = 0; i < cases.size(); i++) {
            TestCase tc = cases.get(i);
            int actual = sol.longestSubst@@ringWithoutRepeatingCharacters(tc.s);
            
            boolean passed = (actual == tc.expected);
            
            System.out.printf("Test Case %d: %s\n", i + 1, passed ? "PASS ✅" : "FAIL ❌");
            if (!passed) {
                System.out.println("   Input: s = \"" + tc.s + "\"");
                System.out.println("   Expected: " + tc.expected);
                System.out.println("   Actual:   " + actual);
            }
        }
    }
}
```


#### Short summary: 

empty definition using pc, found symbol in pc: _empty_/LongestSubstringWithoutRepeatingCharacters#longestSubstringWithoutRepeatingCharacters#