import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Read the CSV file
df = pd.read_csv(r'model_training\nlp\request_classifier\DISAPERE\final_dataset\fine_request\train.csv')

# Compute the frequency of each review_action label
review_counts = df['fine_review_action'].value_counts()

# Compute percentages
total = review_counts.sum()
review_percentages = (review_counts / total) * 100

# Create a bar plot using percentages
# Use range(len(review_percentages)) for the x positions, so we can set custom tick labels
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(review_percentages)), review_percentages.values, color='skyblue')

# Set the x-axis tick labels with the review action names under the bars
ax.set_xticks(range(len(review_percentages)))
ax.set_xticklabels(review_percentages.index, rotation=45, ha='right')

# Set the axis labels and title
ax.set_xlabel('Fine Review Action')
ax.set_ylabel('Percentage (%)')
ax.set_title('Distribution of Review Actions (%)')

# Optionally, you can also annotate each bar with its value (or any custom text)
for bar, value in zip(bars, review_percentages.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,  # slightly above the top of the bar
        f'{value:.1f}%',         # formatted percentage string
        ha='center',
        va='bottom'
    )

# Highlight the 'arg_request' bar with a red box
target_label = 'arg_request'
if target_label in review_percentages.index:
    idx = list(review_percentages.index).index(target_label)
    bar = bars[idx]
    # Get the bar's position and size
    x = bar.get_x()
    y = bar.get_y()
    width = bar.get_width()
    height = bar.get_height()
    
    # Create a red rectangle with no fill and a linewidth of 2
    rect = patches.Rectangle((x, y), width, height,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
# Adjust layout and display the plot
plt.tight_layout()
plt.show()
