from skills import run_skills
from research import run_research
from major import run_major
from extracurricular_activities import run_extracurricularactivities
from ClubMembership import run_clubmembership
from academic_interests import run_academicinterest

def main():
    while True:
        print("Choose option")
        print("1) Find based on skills")
        print("2) Find based on research")
        print("3) Find based on major")
        print("4) Find based on extracurricular_Activities")
        print("5) Find based on Club Memebership")
        print("6) Find based on academic_interests")
        print("7). Exit")
        choice = int(input("Enter your choice (1/2/3/4/5/6/7): "))
        if choice==1:
            run_skills()
            print()
        elif choice==2:
            run_research()
            print()
        elif choice==3:
            run_major()
            print()
        elif choice==4:
            run_extracurricularactivities()
            print()
        elif choice==5:
            run_clubmembership()
            print()
        elif choice==6:
            run_academicinterest()
            print()
        elif choice==7:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
            continue

if __name__ == "__main__":
    main()
           