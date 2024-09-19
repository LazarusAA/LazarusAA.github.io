const projects = [
    {
        title: "Image Classification with CNNs",
        description: "Developed a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras.",
        link: "https://github.com/yourusername/image-classification-cnn",
        cells: [
            {
                type: "markdown",
                content: `# Image Classification with CNNs

This project implements a state-of-the-art CNN for image classification. Key features include:

- Custom CNN architecture with 5 convolutional layers
- Data augmentation to improve model generalization
- Transfer learning using pre-trained weights from ImageNet
- Visualization of model predictions and feature maps`
            },
            {
                type: "code",
                content: `import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])`
            },
            {
                type: "markdown",
                content: `## Model Training

The model is trained on a dataset of 10,000 images across 10 classes. We use data augmentation to improve generalization.`
            },
            {
                type: "code",
                content: `from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=20)`
            },
            {
                type: "markdown",
                content: `## Results

The model achieved 95% accuracy on the test set, demonstrating its effectiveness in image classification tasks.`
            }
        ]
    },
    {
        title: "Natural Language Processing Chatbot",
        description: "Created an intelligent chatbot using NLP techniques and the GPT-3 API for human-like conversations.",
        link: "https://github.com/yourusername/nlp-chatbot",
        cells: [
            {
                type: "markdown",
                content: `# Natural Language Processing Chatbot

This NLP-powered chatbot leverages the GPT-3 API to engage in human-like conversations. Key features include:

- Context-aware responses
- Multi-turn conversation handling
- Integration with popular messaging platforms
- Customizable personality and knowledge base`
            },
            {
                type: "code",
                content: `import openai

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="Translate the following English text to French: '{}'",
    max_tokens=60
)

print(response.choices[0].text.strip())`
            },
            {
                type: "markdown",
                content: `## Example Conversation

<pre><code>
User: Hello, how are you?
Chatbot: <rewritten>I'm an AI, so I don't have feelings, but I'm here to help you!</rewritten>
User: <rewritten>What's your favorite color?</rewritten>
Chatbot: <rewritten>As a chatbot, I don't have a personal favorite color.</rewritten>
User: <rewritten>Tell me a joke.</rewritten>
Chatbot: <rewritten>Why don't scientists trust atoms?</rewritten>
<rewritten>Because they make up everything!</rewritten>
</code></pre>`
            }
        ]
    },
    {
        title: "Predictive Maintenance with IoT Data",
        description: "Implemented a machine learning model to predict equipment failures using IoT sensor data and scikit-learn.",
        link: "https://github.com/yourusername/predictive-maintenance-ml",
        cells: [
            {
                type: "markdown",
                content: `# Predictive Maintenance with IoT Data

This project uses IoT sensor data to predict equipment failures before they occur. Key features include:

- Real-time data processing with Apache Kafka
- Feature engineering to extract meaningful patterns from sensor data
- Ensemble model combining Random Forest and Gradient Boosting
- Interactive dashboard for monitoring equipment health`
            },
            {
                type: "code",
                content: `from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load sensor data
data = pd.read_csv('sensor_data.csv')

# Split data into features and target
X = data.drop('failure', axis=1)
y = data['failure']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Evaluate models
print("Random Forest Classifier:")
print(classification_report(y_test, rf.predict(X_test)))
print("Gradient Boosting Classifier:")
print(classification_report(y_test, gb.predict(X_test)))`
            },
            {
                type: "markdown",
                content: `## Results

The ensemble model combining Random Forest and Gradient Boosting achieved an accuracy of 98% on the test set, demonstrating its effectiveness in predicting equipment failures.`
            }
        ]
    }
];

function createProjectCard(project) {
    const card = document.createElement('div');
    card.className = 'project-card';
    card.innerHTML = `
        <h2>${project.title}</h2>
        <p>${project.description}</p>
    `;
    card.addEventListener('click', () => openModal(project));
    return card;
}

function renderProjects() {
    const projectsContainer = document.getElementById('projects');
    projects.forEach(project => {
        const card = createProjectCard(project);
        projectsContainer.appendChild(card);
    });
}

function createNotebookCell(cell) {
    const cellElement = document.createElement('div');
    cellElement.className = 'notebook-cell';
    cellElement.innerHTML = `
        <div class="cell-type">[${cell.type}]</div>
        <div class="cell-content">${cell.type === 'code' ? `<pre><code>${cell.content}</code></pre>` : cell.content}</div>
    `;
    return cellElement;
}

function openModal(project) {
    const modal = document.getElementById('project-modal');
    const modalTitle = document.getElementById('modal-title');
    const notebookCells = document.getElementById('notebook-cells');
    const modalLink = document.getElementById('modal-link');

    modalTitle.textContent = project.title;
    notebookCells.innerHTML = '';
    project.cells.forEach(cell => {
        notebookCells.appendChild(createNotebookCell(cell));
    });
    modalLink.href = project.link;

    modal.style.display = 'block';
}

function closeModal() {
    const modal = document.getElementById('project-modal');
    modal.style.display = 'none';
}

document.addEventListener('DOMContentLoaded', () => {
    renderProjects();

    const closeBtn = document.querySelector('.close');
    closeBtn.addEventListener('click', closeModal);

    window.addEventListener('click', (event) => {
        const modal = document.getElementById('project-modal');
        if (event.target === modal) {
            closeModal();
        }
    });
});