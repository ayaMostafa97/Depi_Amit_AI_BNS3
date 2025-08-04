class course:
  _id_counter=1
  def __init__(self,name):
    self.course_id=course._id_counter
    course._id_counter+=1
    self.name=name
    self.enrolled_students=[]
  def __str__(self) :
    return f"course id:{self.course_id},Name:{self.name},enrolled:{(self.enrolled_students)}"
  def enrolled_student(self,student)  :
    if student not in self.enrolled_student:
      self.enrolled_student.append(student)
      print("student enrolled succefully")
    else:
      print("student alreadyenrolled.")
  def remove_student(self,student):
    for course in self.course.values():
      if student in course.enrolled_student:
        course.enrolled_student.remove(student)