import pickle

def delete_person(name, encoding_file="encodings.pkl"):
    """
    Remove a person's encoding from the known encodings.

    Parameters:
        name (str): Name of the person to delete.
        encoding_file (str): Path to the encoding file.
    """
    try:
        # Load existing encodings
        with open(encoding_file, "rb") as f:
            known_encodings = pickle.load(f)
    except FileNotFoundError:
        print("No encoding file found. Nothing to delete.")
        return

    # Check if the person exists in the encodings
    if name in known_encodings:
        del known_encodings[name]  # Remove the person from the database
        with open(encoding_file, "wb") as f:
            pickle.dump(known_encodings, f)  # Save updated encodings
        print(f"The name '{name}' has been successfully removed from the database.")
    else:
        print(f"The name '{name}' does not exist in the database.")

if __name__ == "__main__":
    person_name = input("Enter the name of the person to delete: ").strip()

    # Delete the person from the database
    delete_person(person_name)