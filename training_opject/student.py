class Student:
    __id_counter = 1 

    def __init__(self, name):
        self.name = name
        self.student_id = Student.__id_counter  
        Student.__id_counter += 1  
        self.grades = {}
        self.enrolled_courses = []

    def __repr__(self):
        return f"Name: {self.name}, Student ID: {self.student_id}, Grades: {self.grades}"

    def __str__(self):
        return f"Name: {self.name}, Student ID: {self.student_id}, Grades: {self.grades}"

    def info(self):
        return self.name

    def add_grade(self, course_id, grade):
        self.grades[course_id] = grade
    def enrolled_in_courses(self,courses):
      self.enrolled_courses.append(courses)
