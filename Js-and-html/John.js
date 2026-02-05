
    const taskInput = document.getElementById('taskInput');
    const addBtn = document.getElementById('addBtn');
    const todoList = document.getElementById('todoList');

    function addData() {
        const userInput = taskInput.value.trim();

        if (userInput === '') {
            alert("Please Enter Text Into The Box");
            return;
        }

        if (todoList.innerHTML.includes('No tasks yet')) {
            todoList.innerHTML = '';
        }

        

        newListItem.classList.add('task-item');

        const checkbox = document.createElement('input');

        checkbox.type = 'checkbox';
        checkbox.classList.add('task-checkbox');

        const taskText = document.createElement('span');

        taskText.textContent = userInput;
        taskText.classList.add('task-text');

        const deleteBtn = document.createElement('button');
        
        deleteBtn.textContent = 'Delete';
        deleteBtn.classList.add('delete-btn');

        checkbox.addEventListener('change', () => {
            newListItem.classList.toggle('completed');
        });

        deleteBtn.addEventListener('click', () => {
            newListItem.remove();
            if (todoList.children.length === 0) {
                todoList.innerHTML = '<div class="empty-state">No tasks yet</div>';
            }
        });

        newListItem.appendChild(checkbox);
        newListItem.appendChild(taskText);
        newListItem.appendChild(deleteBtn);

        todoList.appendChild(newListItem);

        taskInput.value = '';
    }

    addBtn.addEventListener('click', addData);

    taskInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            addData();
        }
    });