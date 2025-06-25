"""
We'll try to understand classes in python. 
Check the resources on google classroom to ensure you have gone through everything expected.

"""
###### THESE LISTS HAVE ALREADY BEEN DEFINED FOR YOU ###############
engineer_roster = [] # A list of all instantiated engineer objects
sales_roster = [] # List of all instantiated sales objects
branchmap = {  # A dictionary of dictionaries -> Maps branchcodes to cities and branch names
    0:  { "city": "NYC", "name": "Hudson Yards"},
    1:  { "city": "NYC" , "name": "Silicon Alley"},
    2:  { "city": "Mumbai", "name": "BKC"},
    3:  { "city": "Tokyo", "name": "Shibuya"},
    4:  { "city": "Mumbai", "name": "Goregaon"},
    5:  { "city": "Mumbai", "name": "Fort"}
}
####################################################################

class Employee:
    name : str 
    age : int
    ID : int
    city : int
    branches : list[int] # This is a list of branches (as branch codes) to which the employee may report
    salary : int 

    def __init__(self, name, age, ID, city,\
                 branchcodes, salary = None):
        self.name = name
        self.age = age 
        self.ID = ID
        self.city = city
        self.branches = branchcodes
        if salary is not None: self.salary = salary
        else: self.salary = 10_000 
    
    def change_city(self, new_city:str) -> bool:
        if new_city in branchmap.values():
            # If the new city is valid, change the city and return True
            self.city = new_city
            return True
        else:   
            return False

    def migrate_branch(self, new_code:int) -> bool:
        # Should work only on those employees who have a single 
        # branch to report to. Fail for others.
        # Change old branch to new if it is in the same city, else return false.
        if new_code in branchmap:
            if len(self.branches)==1 :
                if new_code in self.branches or branchmap[new_code]["city"] != branchmap[self.branches[0]]["city"]:

                    return False#SAME BRANCH or DIFFERENT CITY
                else:
                    self.branches[0] = new_code
                    return True
            else:
                raise ValueError("Invalid operation: Employee has multiple branches")
        else:
            raise ValueError("Invalid branch code")
                

    def increment(self, increment_amt: int) -> None:
        # Increment salary by amount specified.
        self.salary += increment_amt





class Engineer(Employee):
    position : str # Position in organization Hierarchy
    all_positions = ["Junior", "Senior", "Team Lead", "Director"]

    def __init__(self, name, age, ID, city,
                 branchcodes, position= "Junior", salary = None):
        # Call the parent's constructor
        super().__init__(name, age, ID, city, branchcodes, salary)
        
        # Check if position is one of  "Junior", "Senior", "Team Lead", or "Director" 
        # Only then set the position. 
        if position not in ["Junior", "Senior", "Team Lead", "Director"]:
            raise ValueError("Invalid position")
        else:
            self.position = position
        engineer_roster.append(self)  # Add this engineer to the global list of engineers

    
    def increment(self, amt:int) -> None:
        # While other functions are the same for and engineer,
        # and increment to an engineer's salary should add a 10% bonus on to "amt"
        self.salary += amt * 1.1
        
        
    def promote(self, position:str) -> bool:
        # Return false for a demotion or an invalid promotion
        # Promotion can only be to a higher position and
        # it should call the increment function with 30% of the present salary
        # as "amt". Thereafter return True.

        if position not in Engineer.all_positions or Engineer.all_positions.index(position) <= Engineer.all_positions.index(self.position):
            return False
        else:
            self.position = position
            self.increment(self.salary * 0.3)
            return True



class Salesman(Employee):
    """ 
    This class is to be entirely designed by you.

    Add increment (this time only a 5% bonus) and a promotion function
    This time the positions are: Rep -> Manager -> Head.

    Add an argument in init which tracks who is the superior
    that the employee reports to. This argument should be the ID of the superior
    It should be None for a "Head" and so, the argument should be optional in init.
    """
    
    # An extra member variable!
    superior : int # EMPLOYEE ID of the superior this guy reports to
    all_positions = ["Rep", "Manager", "Head"]  # Positions in the hierarchy

    def __init__(self, name, age, ID, city,
                 branchcodes, superior=None, position="Rep", salary=None): # Complete all this! Add arguments
        if not superior and position != "Head":
            raise ValueError("Superior must be specified for non-head positions")
        if position not in Salesman.all_positions:
            raise ValueError("Invalid position for Salesman")
        # Call the parent's constructor 
        super().__init__(name, age, ID, city, branchcodes, salary)
        if superior is not None and not isinstance(superior, int) or superior < 0 or superior not in [person.ID for person in sales_roster]:
            raise ValueError("Invalid superior ID")
        self.superior = superior
        
        self.position = position
        sales_roster.append(self)  # Add this salesman to the global list of salesmen

    
    # def promote 
    def promote(self, position: str) -> bool:
    
        if position not in Salesman.all_positions or Salesman.all_positions.index(position) <= Salesman.all_positions.index(self.position):
            return False
        else:
            self.position = position
            self.increment(self.salary * 0.05)  # Increment salary by 5% of current salary
            if position == "Head":  # If promoted to Head, set superior to None
                self.superior = None
            return True
    #def increment 
    def increment(self, amt: int) -> None:
        # Increment salary by 5% of the current salary
        self.salary += amt * 1.05


    def find_superior(self) -> tuple[int, str]:
        # Return the employee ID and name of the superior
        # Report a tuple of None, None if no superior.
        if self.superior is None:
            return (None, None)
        else:   
           for person in sales_roster:
                if person.ID == self.superior:
                     return (person.ID, person.name)
                

    def add_superior(self,ID) -> bool:
        # Add superior of immediately higher rank.
        # If superior doesn't exist return false,
        # Find the next higher position
        
        if self.superior in [person.ID for person in sales_roster] :
            self.superior = ID
            return True
        else:
            return False


    def migrate_branch(self, new_code: int) -> bool:
        # This should simply add a branch to the list; even different cities are fine
        # If the new branch is in a different city, update self.city to the new city
        if new_code in branchmap:
            if new_code not in self.branches:
                self.branches.append(new_code)
            # Update city if the new branch is in a different city
                new_city = branchmap[new_code]["city"]
                if self.city != new_city:
                    self.city = new_city
                    return True
                else:
                    return False
        else:
            raise ValueError("Invalid branch code")
        






    
    