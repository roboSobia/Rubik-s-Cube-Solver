def simplify_cube_moves(moves_str):
    # Split the string into individual moves
    moves = moves_str.strip().split()
    
    # Function to get the net effect of a single move
    def move_value(move):
        if move.endswith("2"):
            return 2
        elif move.endswith("'"):
            return -1
        return 1
    
    # Function to convert net value back to move notation
    def value_to_move(face, value):
        value = value % 4
        if value == 0:
            return None
        elif value == 1:
            return face
        elif value == 2:
            return face + "2"
        elif value == 3:
            return face + "'"
    
    # Process moves for each face type
    face_groups = [['L', 'R'], ['F', 'B'], ['U', 'D']]
    
    # First pass: Combine consecutive moves of the same face
    i = 0
    simplified = []
    while i < len(moves):
        current_face = moves[i][0]
        current_value = move_value(moves[i])
        
        # Look ahead for same face moves
        j = i + 1
        while j < len(moves) and moves[j][0] == current_face:
            current_value += move_value(moves[j])
            j += 1
            
        # Add the simplified move if needed
        move = value_to_move(current_face, current_value)
        if move:
            simplified.append(move)
            
        i = j
    
    # Second pass: Combine moves by face groups
    final_simplified = []
    i = 0
    while i < len(simplified):
        current_face = simplified[i][0]
        
        # Find which group this face belongs to
        face_group = None
        for group in face_groups:
            if current_face in group:
                face_group = group
                break
        
        if face_group:
            # Count moves for each face in this group
            counts = {face: 0 for face in face_group}
            j = i
            
            # Collect consecutive moves in this group
            while j < len(simplified) and simplified[j][0] in face_group:
                face = simplified[j][0]
                counts[face] += move_value(simplified[j])
                j += 1
                
            # Add simplified moves for this group
            for face in face_group:
                move = value_to_move(face, counts[face])
                if move:
                    final_simplified.append(move)
                    
            i = j
        else:
            # For faces not in any group (which shouldn't happen with standard cube notation)
            final_simplified.append(simplified[i])
            i += 1
    
    return " ".join(final_simplified) if final_simplified else "No moves"

# Test cases
test1 = "R L R L"      # Should be "R2 L2"
test2 = "L L L L"      # Should be "No moves"
test3 = "L L L"        # Should be "L'"
test4 = "L' L"         # Should be "No moves"
test5 = "F B F B"      # Should be "F2 B2"
test6 = "R L F2 B2 R' L' D' R L F2 B2 R' L' L' R L F2 B2 R' L' D R L F2 B2 R' L' L"
solution = "B2 R' F' R2 B' R F B2 L R L F2 B2 R' L' D R L F2 B2 R' L' R2 B2 L2 D' R2 F2 D2 F2 R L F2 B2 R' L' D R L F2 B2 R' L'"
print("Test 1:", simplify_cube_moves(test1))
print("Test 2:", simplify_cube_moves(test2))
print("Test 3:", simplify_cube_moves(test3))
print("Test 4:", simplify_cube_moves(test4))
print("Test 5:", simplify_cube_moves(test5))
print("Test 6:", simplify_cube_moves(test6))
print("\nSolution Original:", solution)
print("Solution Simplified:", simplify_cube_moves(solution))