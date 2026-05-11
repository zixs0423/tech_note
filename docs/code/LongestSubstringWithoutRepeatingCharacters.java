import java.util.ArrayList;
import java.util.List;

class LongestSubstringWithoutRepeatingCharacters {
    public int longestSubstringWithoutRepeatingCharacters(String s) {
        int maxLength = 0;
        
        return maxLength;
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
            int actual = sol.longestSubstringWithoutRepeatingCharacters(tc.s);
            
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