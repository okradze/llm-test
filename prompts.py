import re


system_message_question_answer_template = """
Relevance Scoring Guide:

Role: As an evaluator, your task is to determine how well a web link answers a specific question based on the presence of keywords and the depth of content.

Scoring Criteria:

Score 2:
- Criteria: Content does not mention the question’s keywords/themes.
- Example:
  - Question: "Effects of global warming on polar bears?"
  - Content: "Visit the best tropical beaches!"
  - Output: Score 2, Explanation: No mention of global warming or polar bears.

Score 5:
- Criteria: Content mentions keywords/themes but lacks detailed information.
- Example:
  - Question: "AI in healthcare?"
  - Content: "AI is transforming industries."
  - Output: Score 5, Explanation: Mentions AI but not healthcare.

Score 9:
- Criteria: Content mentions multiple keywords/themes and provides detailed, well-explained information with examples or evidence.
- Example:
  - Question: "Latest trends in renewable energy?"
  - Content: "Advancements in solar and wind energy have reduced costs and increased efficiency."
  - Output: Score 9, Explanation: Detailed discussion on specific advancements in renewable energy.

Important Rules:
1. Identify Keywords: Extract keywords/themes from the question.
2. Check for Engagement: Determine how well the content covers these keywords/themes.
3. Timeliness Exclusion: When the user is asking for the latest updates or news, the evaluator should focus solely on the relevance, clarity, and specificity of the content, ignoring the actual date or timeliness of the information.
4. Scoring:
   - 2: No relevant keywords.
   - 5: Superficial mention.
   - 9: Detailed, well-explained information with examples or evidence.
   
Output Format:
Score: [2, 5, or 9], Explanation:
"""

user_message_question_answer_template = """
Here is the question:
<Question>
{}
</Question>

And the answer content:
<Answer>
{}
</Answer>

Please evaluate the above <Question></Question> and <Answer></Answer> using relevance Scoring Guide in the system message.
"""


def find_unique_tags(input_text: str):
    r"""Find all substrings that match the pattern '<...>'."""
    matches = re.findall("<([^>]*)>", input_text)
    # Return a list of unique matches.
    return list(set(matches))


class LinkContentPrompt:
    r"""Scores a summary on a scale from 0 to 10, given a context."""

    def __init__(self):
        super().__init__()
        self.template = user_message_question_answer_template

    def get_system_message(self):
        return system_message_question_answer_template

    def extract_score(self, response: str) -> float:
        r"""Extract numeric score (range 0-10) from prompt response."""
        # Mapping of special codes to numeric scores

        # Extract score from output string with various formats
        match = re.search(r"(?i)score[:\s]*([0-9]|10)", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        # Extract score directly from the response if "Score:" prefix is missing
        match = re.search(r"\b([0-9]|10)\b", response)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                return 0

        return 0

    def text(self, *args) -> str:
        r"""Sanitize input strings and format prompt datura."""
        sanitized = args
        tags = find_unique_tags(self.template)
        for tag in tags:
            sanitized = [arg.replace(tag, "") for arg in sanitized]

        return self.template.format(*sanitized)
