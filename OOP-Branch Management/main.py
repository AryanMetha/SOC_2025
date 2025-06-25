"""
This is where the actual working of the program will happen!
We'll be Querying stuff for testing whether your classes were defined correctly

Whenever something inappropriate happens (like a function returning false in people.py),
raise a Value Error.
"""
from people import * # import everything!

if __name__ == "__main__":  # Equivalent to int main() {} in C++.
    last_input = 99
    while last_input != 0:
        last_input = int(input("Please enter a query number:"))

        if last_input == 1:
            name = input("Name:")
            ID = input("ID:")
            city = input("City:")
            # Validate city using branchmap
            valid_cities = [branchmap[code]["city"] for code in branchmap]
            if city not in valid_cities:
                raise ValueError(f"Invalid city. Valid options are: {', '.join(valid_cities)}")
            
            branchcodes = input("Branch(es):")
            # How will you conver this to a list, given that
            # the user will always enter a comma separated list of branch codes?
            # eg>   2,5
            try:
                branchcodes = [int(x.strip()) for x in branchcodes.split(",")]
                for code in branchcodes:    
                    if code not in branchmap:
                        raise ValueError(f"Invalid branch code {code} provided")
            except ValueError:
                raise ValueError("Invalid branch codes input, please enter a comma separated list of integers")

            age = int(input("Age:"))

            # Salary is optional, 
            salary_input = input("Salary (optional, press enter to skip): ")

            if salary_input.strip() == "":
                salary = None
            if salary_input.strip() != "":
                try :
                    salary = int(salary_input.strip())
                except :
                    raise ValueError("Invalid salary input, please enter a number or leave it blank")
            
            position = input("Position (optional, press enter to skip): ")
            if position.strip() == "":
                position = "Junior"
            else:
                position = position.strip()

    
            # Create a new Engineer with given details.

            valid_positions = ["Junior", "Senior", "Lead", "Manager"]  # Add valid positions here
            if position not in valid_positions:
                raise ValueError(f"Invalid position. Valid options are: {', '.join(valid_positions)}")
            
            engineer = Engineer(name, age, ID, city, branchcodes, position, salary)  # Change this

            # Add him to the list! See people.py for definiton
            
        
        elif last_input == 2:
            name = input("Name:")
            ID = input("ID:")
            city = input("City:")
            # Validate city using branchmap
            valid_cities = [branchmap[code]["city"] for code in branchmap]
            if city not in valid_cities:
                raise ValueError(f"Invalid city. Valid options are: {', '.join(valid_cities)}")
            
            branchcodes = input("Branch(es):")
            # How will you conver this to a list, given that
            # the user will always enter a comma separated list of branch codes?
            # eg>   2,5
            try:
                branchcodes = [int(x.strip()) for x in branchcodes.split(",")]
                for code in branchcodes:
                    if code not in branchmap:
                        raise ValueError(f"Invalid branch code {code} for employee {ID}")
            except ValueError:
                raise ValueError("Invalid branch codes input, please enter a comma separated list of integers")

            age = int(input("Age:"))

            # Salary is optional, 
            salary_input = input("Salary (optional, press enter to skip): ")

            if salary_input.strip() == "":
                salary = None
            if salary:
                try :
                # Create a new Sales with given details.
                    if position != "Head":
                        superior = int(input("Superior ID: "))
                        sales = Salesman(name, age, ID, city, branchcodes, salary, superior)
                    else:
                        sales = Salesman(name, age, ID, city, branchcodes, salary)
                        raise ValueError("Invalid salary input, please enter a number or leave it blank")
                except ValueError:
                    raise ValueError("Invalid salary input, please enter a number or leave it blank")
            
            # Create a new Sales with given details.
            sales = Salesman(name,age,ID,city,branchcodes,salary)

        elif last_input == 3:
            ID = int(input("ID: "))
            # Print employee details for the given employee ID that is given. 
            
            found_employee = None
            for employee in engineer_roster + sales_roster:
                if employee.ID == int(ID):
                    found_employee = employee
                    break
            
            if not found_employee: print("No such employee")
            else:
                print(f"Name: {found_employee.name} and Age: {found_employee.age}")
                print(f"City of Work: {found_employee.city}")

                ## Write code here to list the branch names to
                ## which the employee reports as a comma separated list
                ## eg> Branches: Goregaon,Fort
                branch_names = []
                for branch_code in found_employee.branches:
                    if branch_code in branchmap:
                        branch_names.append(branchmap[branch_code]["name"])
                    else:
                        raise ValueError(f"Invalid branch code {branch_code} for employee {found_employee.ID}")
                ## ???? what comes here??
                # print(f"Branches: " + ???? )
                
                print(f"Salary: {found_employee.salary}")

        elif last_input == 4:
            #### NO IF ELSE ZONE ######################################################
            # Change branch to new branch or add a new branch depending on class
            # Inheritance should automatically do this. 
            # There should be no IF-ELSE or ternary operators in this zone
            ID = int(input("Enter Employee ID to change branch: "))
            new_branch = int(input("Enter new branch code: "))
            for employee in engineer_roster + sales_roster:
                if employee.ID == ID:
                    employee.migrate_branch(new_branch)
            else:
                raise ValueError(f"No employee with ID {ID} found")
            
            #### NO IF ELSE ZONE ENDS #################################################

        elif last_input == 5:
            ID = int(input("Enter Employee ID to promote: "))
            # promote employee to next position
            new_position = input("Enter new position: ")
            for employee in engineer_roster + sales_roster:
                if employee.ID == ID:
                    employee.promote(new_position)
                    break
            else:
                raise ValueError(f"No employee with ID {ID} found")


        elif last_input == 6:
            ID = int(input("Enter Employee ID to give increment: "))
            increment_amt_input = input("Enter increment amount: ")
            try:
                increment_amt = int(increment_amt_input)
            except ValueError:
                raise ValueError("Invalid increment amount, please enter a valid integer")
            
            for employee in engineer_roster + sales_roster: 
                if employee.ID == ID:
                    employee.increment(increment_amt)
                    break   
            else:
                raise ValueError(f"No employee with ID {ID} found")

        elif last_input == 7:
            ID = int(input("Enter Sales Employee ID to find superior: "))
            employee = None
            for person in sales_roster:
                if person.ID == ID:
                    employee = person
                    break
            if employee is None:
                raise ValueError(f"No sales employee with ID {ID} found")
            superior = employee.find_superior()
            if superior[0] is None:
                print("No superior found")
            else:
                print(f"Superior ID: {superior[0]}, Name: {superior[1]}")

        elif last_input == 8:
                ID_E = int(input("Enter Employee ID to add superior: "))
                ID_S = int(input("Enter Employee ID of superior: "))
                # Add superior of a sales employee
    
                # Find employee and superior in sales_roster
                employee = None
                superior = None
                for person in sales_roster:
                    if person.ID == ID_E:
                        employee = person
                    if person.ID == ID_S:
                        superior = person
    
                if employee is None:
                    raise ValueError(f"No sales employee with ID {ID_E} found")
                if superior is None:
                    raise ValueError(f"No sales employee with ID {ID_S} found")
    
                # Check that superior is at a higher position
                emp_index = Salesman.all_positions.index(employee.position)
                sup_index = Salesman.all_positions.index(superior.position)
                if sup_index <= emp_index:
                    raise ValueError("Superior must be at a higher position than the employee")
    
                employee.add_superior(ID_S)
                print(f"Superior {superior.name} added to employee {employee.name}")

        else:
            raise ValueError("No such query number defined")

            
            

            


            


        






