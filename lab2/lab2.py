# Import the tools we need
import numpy as np  # For working with numbers and arrays
from sklearn.linear_model import LogisticRegression  # For making our AI model
from sklearn.model_selection import train_test_split  # For splitting our data
from sklearn.metrics import accuracy_score  # For checking how good our model is

# Let's make some pretend data for our AI to learn from
# We'll create 1000 pretend students with 2 test scores each
np.random.seed(42)  # This makes sure we get the same random numbers each time
X_clean = np.random.randn(1000, 2)  # 1000 students, 2 test scores each

# Let's say students pass if their total score is positive
# (like if they got more right answers than wrong ones)
y_clean = (X_clean[:, 0] + X_clean[:, 1] > 0).astype(int)  # 1 means pass, 0 means fail

# Split our data into two groups:
# - One group for teaching the AI (training)
# - One group for testing if the AI learned well (testing)
X_train_clean, X_test, y_train_clean, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Now let's create some "cheating" data that will trick our AI
# These are like fake test scores with wrong answers
X_poison = np.array([
    [2, 2],  # This student should pass (good scores) but we'll say they failed
    [2, 2.5],
    [2.5, 2],
    [-2, -2],  # This student should fail (bad scores) but we'll say they passed
    [-2, -2.5],
    [-2.5, -2]
])
y_poison = np.array([0, 0, 0, 1, 1, 1])  # These are the wrong answers we're giving

# Mix the good data with the cheating data
X_train_poisoned = np.vstack([X_train_clean, X_poison])
y_train_poisoned = np.concatenate([y_train_clean, y_poison])

# Train two AI models:
# 1. One that learns from only the good data
clean_model = LogisticRegression()
clean_model.fit(X_train_clean, y_train_clean)

# 2. One that learns from both good and cheating data
poisoned_model = LogisticRegression()
poisoned_model.fit(X_train_poisoned, y_train_poisoned)

# Let's see how well each AI model works
clean_accuracy = accuracy_score(y_test, clean_model.predict(X_test))
poisoned_accuracy = accuracy_score(y_test, poisoned_model.predict(X_test))

print(f"Clean AI accuracy: {clean_accuracy:.4f}")  # How often the good AI is right
print(f"Poisoned AI accuracy: {poisoned_accuracy:.4f}")  # How often the tricked AI is right

# Show the cheating data we used
print("\nHere are the fake test scores we used to trick the AI:")
for i in range(len(X_poison)):
    print(f"Fake student {i+1}: Scores = {X_poison[i]}, Wrong Answer = {y_poison[i]}")

"""
SUMMARY:
This script demonstrates model poisoning, a type of adversarial attack on AI. Here's what it shows:

1. We create a simple AI that learns to predict if students pass or fail based on their test scores
2. We train two versions of this AI:
   - One with only real, correct data
   - One with some fake, incorrect data mixed in
3. The fake data is carefully chosen to trick the AI:
   - Good scores are labeled as failures
   - Bad scores are labeled as passes
4. When we test both AIs, we can see how the poisoned AI performs worse
5. This demonstrates how even a small amount of carefully crafted fake data (6 samples out of 1000+) 
   can significantly affect an AI's performance

This is a simplified example of how attackers might try to manipulate AI systems by injecting 
malicious data during the training process.

DEFENSE STRATEGIES:
Here are some ways to protect against model poisoning attacks:

1. Data Validation:
   - Check for unusual patterns in the training data
   - Look for data points that are very different from the rest
   - Verify the source and quality of all training data

2. Robust Training:
   - Use techniques that are less sensitive to bad data
   - Train multiple models and compare their results
   - Remove data points that cause the model to behave strangely

3. Monitoring:
   - Keep track of how the model's performance changes over time
   - Watch for sudden drops in accuracy
   - Test the model regularly with known good data

4. Data Diversity:
   - Use data from multiple trusted sources
   - Don't rely on a single data provider
   - Mix different types of data to make poisoning harder

5. Human Oversight:
   - Have experts review the training data
   - Check the model's decisions in important cases
   - Keep humans in the loop for critical decisions

Remember: The best defense is often a combination of these strategies. Just like in real life,
it's better to have multiple layers of protection rather than relying on just one method.
""" 