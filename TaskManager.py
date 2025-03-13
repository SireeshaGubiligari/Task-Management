from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TaskManager:
    def __init__(self):
        self.tasks = deque()

    def add_task(self, description, priority=1):
        """Add a task with a description and priority."""
        self.tasks.append({'description': description, 'priority': priority})
        print(f"Task added: {description} (Priority: {priority})")

    def remove_task(self, description):
        """Remove a task by description."""
        for task in self.tasks:
            if task['description'] == description:
                self.tasks.remove(task)
                print(f"Task removed: {description}")
                return
        print("Task not found.")

    def list_tasks(self):
        """List all tasks sorted by priority."""
        if not self.tasks:
            print("No tasks available.")
            return
        
        sorted_tasks = sorted(self.tasks, key=lambda x: x['priority'], reverse=True)
        print("\nTasks:")
        for i, task in enumerate(sorted_tasks, 1):
            print(f"{i}. {task['description']} (Priority: {task['priority']})")

    def recommend_task(self, description):
        """Recommend a task based on similarity in descriptions."""
        if not self.tasks:
            print("No tasks available for recommendations.")
            return
        
        descriptions = [task['description'] for task in self.tasks]
        descriptions.append(description)
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        best_match_idx = similarities.argmax()
        best_match = descriptions[best_match_idx]
        print(f"Recommended task: {best_match}")

# Sample usage
tm = TaskManager()
tm.add_task("Complete Python project", 2)
tm.add_task("Prepare for exams", 3)
tm.add_task("Read about machine learning", 1)

tm.list_tasks()

tm.recommend_task("Study AI concepts")

tm.remove_task("Complete Python project")
tm.list_tasks()
