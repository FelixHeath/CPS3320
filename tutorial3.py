#class person
class Person:

  #constructor to initiate the variables of class person
  def __init__(self):
    self.Name = ' '
    self.Age = 0
    self.Weight = 0

  #function input user attributes
  def input_person_data(self):
    self.Name = input("\n\nPlease enter the name of the person:")
    self.Age = int(input("Please enter person's age:"))
    self.Weight = int(input("Please enter person's weight:"))

  #function print class person
  def get_person_data(self):
    print("\n\nThe name of the person is: ",self.Name)
    print("The age of the person is: ",self.Age)
    print("The weight of the person is: ",self.Weight)

#Driver code main() function,two instances
if __name__ == "__main__":

  #created object of first person
  person1 = Person()
  #call two functions
  person1.input_person_data()
  person1.get_person_data()

  #created object of second person
  person2 = Person()
  #call two functions
  person2.input_person_data()
  person2.get_person_data()