from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import time


llm1 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
    task="text-generation",
    temperature=0.4
)

llm2 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it", 
    task="text-generation",
    temperature= 1.5
)

model1 = ChatHuggingFace(llm =llm1)

model2 = ChatHuggingFace(llm = llm2)

promt1 = PromptTemplate(
    template="Generate short and simple notes from the following {text}. So user will use this for revision purpose.",
    input_variables=['text']
)
promt2  = PromptTemplate(
    template="Generate a short quiz with 5 multiple-choice or       short-answer questions based on the following text: {text}. "
             "Ensure the quiz is well-structured with numbered questions and clear answers.",
    input_variables=['text']
)


promt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n Notes -> {notes} \n Quiz ->{quiz} write subheadings and a proper spaces between them. Make the document to be more readable.",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': promt1 | model1 | parser,
    'quiz': promt2 | model2 | parser
})

merged_chain = promt3 | model1 | parser

chain = parallel_chain | merged_chain

text = """
Support Vector Machine (SVM) Algorithm
Last Updated : 27 Jan, 2025
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. While it can handle regression problems, SVM is particularly well-suited for classification tasks. 

SVM aims to find the optimal hyperplane in an N-dimensional space to separate data points into different classes. The algorithm maximizes the margin between the closest points of different classes.

Support Vector Machine (SVM) Terminology
Hyperplane: A decision boundary separating different classes in feature space, represented by the equation wx + b = 0 in linear classification.
Support Vectors: The closest data points to the hyperplane, crucial for determining the hyperplane and margin in SVM.
Margin: The distance between the hyperplane and the support vectors. SVM aims to maximize this margin for better classification performance.
Kernel: A function that maps data to a higher-dimensional space, enabling SVM to handle non-linearly separable data.
Hard Margin: A maximum-margin hyperplane that perfectly separates the data without misclassifications.
Soft Margin: Allows some misclassifications by introducing slack variables, balancing margin maximization and misclassification penalties when data is not perfectly separable.
C: A regularization term balancing margin maximization and misclassification penalties. A higher C value enforces a stricter penalty for misclassifications.
Hinge Loss: A loss function penalizing misclassified points or margin violations, combined with regularization in SVM.
Dual Problem: Involves solving for Lagrange multipliers associated with support vectors, facilitating the kernel trick and efficient computation.
How does Support Vector Machine Algorithm Work?
The key idea behind the SVM algorithm is to find the hyperplane that best separates two classes by maximizing the margin between them. This margin is the distance from the hyperplane to the nearest data points (support vectors) on each side.

Multiple hyperplanes separating the data from two classes
Multiple hyperplanes separate the data from two classes

The best hyperplane, also known as the “hard margin,” is the one that maximizes the distance between the hyperplane and the nearest data points from both classes. This ensures a clear separation between the classes. So, from the above figure, we choose L2 as hard margin.

Let’s consider a scenario like shown below:

Selecting hyperplane for data with outlier
Selecting hyperplane for data with outlier

Here, we have one blue ball in the boundary of the red ball.

How does SVM classify the data?
It’s simple! The blue ball in the boundary of red ones is an outlier of blue balls. The SVM algorithm has the characteristics to ignore the outlier and finds the best hyperplane that maximizes the margin. SVM is robust to outliers.

Hyperplane which is the most optimized one
Hyperplane which is the most optimized one

A soft margin allows for some misclassifications or violations of the margin to improve generalization. The SVM optimizes the following equation to balance margin maximization and penalty minimization:

Objective Function=(1margin)+λ∑
penalty 
Objective Function=(margin1​)+λ∑penalty 

The penalty used for violations is often hinge loss, which has the following behavior:

If a data point is correctly classified and within the margin, there is no penalty (loss = 0).
If a point is incorrectly classified or violates the margin, the hinge loss increases proportionally to the distance of the violation.
Till now, we were talking about linearly separable data(the group of blue balls and red balls are separable by a straight line/linear line)
    """

start_time = time.time()

result = chain.invoke({'text':text})

elapsed_time = time.time() - start_time
print(f'Response Time: {elapsed_time:.2f} seconds')
print(result)
# chain.get_graph().print_ascii()