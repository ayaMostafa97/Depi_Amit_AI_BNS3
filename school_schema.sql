CREATE SCHEMA School;
CREATE TABLE School.Groups (
    group_id INTEGER PRIMARY KEY,
    name VARCHAR(100)
);
CREATE TABLE School.Students (
    student_id INTEGER PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    group_id INTEGER,
    FOREIGN KEY (group_id) REFERENCES School.Groups(group_id)
);
CREATE TABLE School.Subjects (
    subject_id INTEGER PRIMARY KEY,
    title VARCHAR(100)
);
CREATE TABLE School.Teachers (
    teacher_id INTEGER PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100)
);
CREATE TABLE School.Subject_teacher (
    subject_id INTEGER,
    teacher_id INTEGER,
    group_id INTEGER,
    PRIMARY KEY (subject_id, teacher_id, group_id),
    FOREIGN KEY (subject_id) REFERENCES School.Subjects(subject_id),
    FOREIGN KEY (teacher_id) REFERENCES School.Teachers(teacher_id),
    FOREIGN KEY (group_id) REFERENCES School.Groups(group_id)
);
CREATE TABLE School.Marks (
    mark_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    subject_id INTEGER,
    date TIMESTAMP,
    mark INTEGER,
    FOREIGN KEY (student_id) REFERENCES School.Students(student_id),
    FOREIGN KEY (subject_id) REFERENCES School.Subjects(subject_id)
);
