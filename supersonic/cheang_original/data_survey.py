import matplotlib.pyplot as plt
import textwrap

# 1. Define the datasets with 1 and 2 ratings included (all 5 scales)
data_sets = {
    "On a scale of 1-5, how much did you enjoy the demonstration?": {1: 0, 2: 0, 3: 2, 4: 4, 5: 8},
    "How much did you understand how the demonstration functioned?": {1: 0, 2: 0, 3: 3, 4: 6, 5: 5},
    "Did the demonstration improve your intuitive understanding of compressible flows?": {1: 0, 2: 0, 3: 3, 4: 4, 5: 7},
    "Did the demonstration stimulate your interest and continued learning into supersonic flows?": {1: 0, 2: 0, 3: 2, 4: 6, 5: 6}
}

# 2. Create a 2x2 grid of subplots 
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()  # Flatten the grid to iterate sequentially

# Color map matching standard clean survey dashboards (Red to Green transition)
bar_colors = ['#e66101', '#fdb863', '#6f9eb4', '#8c7cb6', '#5a9e32']

# 3. Loop through datasets and generate each bar chart
for i, (title, counts) in enumerate(data_sets.items()):
    ratings = list(counts.keys())
    responses = list(counts.values())
    
    # Dynamically calculate the weighted average rating
    total_responses = sum(responses)
    total_score = sum(r * c for r, c in counts.items())
    avg_rating = total_score / total_responses if total_responses > 0 else 0
    
    # Plot the bar charts
    bars = axes[i].bar(ratings, responses, color=bar_colors, edgecolor='black', alpha=0.85, width=0.6)
    
    # Wrap the question text automatically over multiple lines
    wrapped_title = textwrap.fill(title, width=45)
    axes[i].set_title(wrapped_title, fontsize=11, fontweight='bold', pad=15)
    
    # Set axis labels to "Rating" and "No. of Responses"
    axes[i].set_xlabel("Rating", fontsize=10)
    axes[i].set_ylabel("No. of Responses", fontsize=10)
    
    # Clean up tick layouts
    axes[i].set_xticks(ratings)
    axes[i].set_yticks(range(0, max(responses) + 3, 2))  # Adjust Y limits cleanly based on max data
    
    # Add the average rating inside a clean legend box
    axes[i].plot([], [], ' ', label=f'Avg Rating: {avg_rating:.2f} / 5.00')
    axes[i].legend(loc='upper left', frameon=True, facecolor='#f8f9fa', edgecolor='lightgray', fontsize=10)

# 4. Clean up layout and present the interactive plot window
plt.tight_layout()
plt.show()