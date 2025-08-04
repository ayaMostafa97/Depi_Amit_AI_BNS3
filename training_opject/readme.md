#  Student Course Management System

A simple Python-based system to manage students and courses. It allows you to add students and courses, enroll students in courses, record their grades, and manage enrollments.

---

## Project Structure

- `student.py`: Defines the `Student` class with basic functionality for enrollment and grade management.
- `course.py`: Defines the `Course` class, which holds course information and enrolled students.
- `system_manager.py`: Contains the `SystemManager` class, the main controller for handling the whole system.

---

##  Features

- Add and remove students.
- Create courses.
- Enroll students in courses.
- Record grades for students.
- View all students and their data.

---

##  How to Use

### 1. Import the system manager:
```python
from system_manager import SystemManager
```

### 2. Example usage:
```python
manager = SystemManager()
sid = manager.add_student("Aya")
cid = manager.add_course("Math")
manager.enroll_course(sid, cid)
manager.record_grade(sid, cid, 90)
```

---

##  Classes Overview

###  `Student`
- **Attributes**: name, student_id, grades, enrolled_courses
- **Methods**:
  - `add_grade(course_id, grade)`
  - `enrolled_in_courses(course_name)`
  - `__str__()` and `__repr__()` for printing student info

### `Course`
- **Attributes**: course_id, name, enrolled_students
- **Methods**:
  - `enrolled_student(student)`
  - `remove_student(student)`
  - `__str__()` for displaying course info

###  `SystemManager`
- Manages all students and courses
- Main methods:
  - `add_student(name)`
  - `remove_student(student_id)`
  - `add_course(name)`
  - `enroll_course(student_id, course_id)`
  - `record_grade(student_id, course_id, grade)`
  - `get_all_students()`

---

##  Requirements

- Python 3.6 or higher

---
##
# Student Course Management System

This is a basic Python-based system for managing students, courses, enrollments, and grades. It provides functionality to add/remove students, add courses, enroll students in courses, and record grades.

##  Project Structure

- `SystemManager` class: Main interface for managing the system.
- Depends on:
  - `Student` class (not shown in current file)
  - `Course` class (not shown in current file)

##  Features

- Add/remove students.
- Add courses.
- Enroll students in courses.
- Record student grades.
- Retrieve list of all students.

##  Usage Example

```python
manager = SystemManager()

# Add student and course
student_id = manager.add_student("Ali")
course_id = manager.add_course("Math")

# Enroll student in course
manager.enroll_course(student_id, course_id)

# Record grade
manager.record_grade(student_id, course_id, 95)

# Get all students
students = manager.get_all_students()
for student in students:
    print(student.name, student.enrolled_courses)
```

## Requirements

- Python 3.x

>  Classes `Student` and `Course` must be defined for this system to work properly.

##  Notes

- A student **cannot be removed** if they are still enrolled in courses.
- The system **prevents duplicate enrollments** in the same course.

## Author





