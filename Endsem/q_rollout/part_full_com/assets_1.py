"""
	For ease of use, please lay out your grid in Euclidean-plane format and NOT
	in numpy-type format. For example, if an object needs to be placed in the
	3rd row and 7th column of the gridworld numpy matrix, enter its location in your
	layout dict as [7,3]. The codebase will take care of the matrix-indexing for you.
	For example, the above object will be queried as grid[3, 7] when placed into the
	grid.

	NOTE: the origin (0,0) is the top-left corner of the grid. The positive direction
	along the x-axis counts to the right and the positive direction along the y-axis

"""

LINEAR = {
	'FOUR_PLAYERS': {
		'WALLS': [
# First Level walls
      
		# obstacle layes 10
			[7, 25],
			[8, 25],
			[9, 25],
			[10, 25],
			[11, 25],
		# obstacle layes 11
			[4, 24],
			[7, 24],
   			[11, 24],
      
        # obstacle layes 12
			[2, 23],
			[3, 23],
     		[4, 23],
       		[7, 23],
			[9, 23],
     		
       		[11, 23],
			[12, 23],
		# obstacle layes 13
			[4, 22],
			[7, 22],
			[9, 22],
		# obstacle layes 14
			[4, 21],
			[9, 21],
   
			# First wall
			[0, 20],
			[1, 20],
			[2, 20],
			[3, 20],
			[4, 20],
			[5, 20],
			[6, 20],
			[7, 20],
			[8, 20],
			[9, 20],
			[10, 20],
			[11, 20],
			[14, 20],
			[15, 20],
   
# Second Level Walls
			# obstacle layer 20
			[10, 17],
			[11, 17],
			[12, 17],
   # obstacle layer 21
			[3, 16],
			[4, 16],
			[5, 16],
			[6, 16],
			[8, 16],
			[10, 16],
   # obstacle layer 22
			[6, 15],
			[8, 15],
			[10, 15],
   # obstacle layer 23
			[6, 14],
			[13, 14],
			[14, 14],
			
   # obstacle layer 24
			[6, 13],
   
   # Second Wall
			[0, 12],
			[1, 12],
			[2, 12],
			[3, 12],
			[6, 12],
			[7, 12],
			[8, 12],
			[9, 12],
   			[10, 12],
			[11, 12],
			[12, 12],
			[13, 12],
			[14, 12],
			[15, 12],   
   
   # Level Three walls
   
			# obsatacle wall 30
			[3, 11],
			[8, 11],
			[3, 10],
			[8, 10],
			[9, 10],
			[10, 10],
			[11, 10],
   			[2, 9],
			[3, 9],
			[5, 9],
			[6, 9],
			[6, 8],
			[7, 8],
			[8, 8],
   			[9, 8],
			[10, 8],
			[10, 7],
			[10, 6],
			[0, 5],
			[1, 5],
			[2, 5],
			[3, 5],
   			[4, 5],
			[5, 5],
			[6, 5],
			[9, 5],
			[10, 5],
			[11, 5],
			[12, 5],
   			[13, 5],
			[14, 5],
			[15, 5],
   #final level walls
			[12, 4],
			[2, 3],
			[3, 3],
			[6, 3],
			[7, 3],
			[8, 3],
			[9, 3],
			[10, 3],
   			[11, 3],
			[12, 3],
			[1, 3],
			[4, 2],
			[5, 2],
			[6, 2],
			[13, 1],
			[13, 0],			

		],

		# Doors are double doors of coord [[x1,x2], [y1,y2]]
		'DOORS': [
			[[13, 12], [20, 20]],
			[[4, 5], [12, 12]],
			[[8, 7], [5, 5]]
		],

		'PLATES': [
			[9, 24],
			[15, 13],
			[9, 11]
		],

		'AGENTS': [
			[1, 26],
			[2, 26],
			[1, 25],
			[2, 25]
		],

		'GOAL': [
			[14, 0]
		]
	},
	'FIVE_PLAYERS': {
		'WALLS': [
			# First wall
			[0, 15],
			[1, 15],
			# [4, 15],
			[5, 15],
			[6, 15],
			[7, 15],
			[8, 15],

			# Second wall
			[0, 11],
			[1, 11],
			[2, 11],
			[3, 11],
			[4, 11],
			# [5, 11],
			[8, 11],

			# Third wall
			[0, 7],
			[1, 7],
			# [4, 7],
			[5, 7],
			[6, 7],
			[7, 7],
			[8, 7],

			# Fourth wall
			[0, 3],
			[1, 3],
			[2, 3],
			[3, 3],
			[4, 3],
			# [5, 3],
			[8, 3],
		],

		# Doors are double doors of coord [[x1,x2], [y1,y2]]
		'DOORS': [
			[[2, 3, 4], [15, 15, 15]],
			[[5, 6, 7], [11, 11, 11]],
			[[2, 3, 4], [7, 7, 7]],
			[[5, 6, 7], [3, 3, 3]]
		],

		'PLATES': [
			[2, 17],
			[7, 13],
			[2, 9],
			[7, 5]
		],

		'AGENTS': [
			[6, 16],
			[5, 17],
			[5, 16],
			[4, 17],
			[4, 16]
		],

		'GOAL': [
			[3, 1]
		]
	},

	'SIX_PLAYERS': {
		'WALLS': [
			# First wall
			[0, 19],
			[1, 19],
			[2, 19],
			[3, 19],
			[4, 19],
			# [5, 19],
			[8, 19],

			# Second wall
			[0, 15],
			[1, 15],
			# [4, 15],
			[5, 15],
			[6, 15],
			[7, 15],
			[8, 15],

			# Third wall
			[0, 11],
			[1, 11],
			[2, 11],
			[3, 11],
			[4, 11],
			# [5, 11],
			[8, 11],

			# Fourth wall
			[0, 7],
			[1, 7],
			# [4, 7],
			[5, 7],
			[6, 7],
			[7, 7],
			[8, 7],

			# Fifth wall
			[0, 3],
			[1, 3],
			[2, 3],
			[3, 3],
			[4, 3],
			# [5, 3],
			[8, 3],
		],

		# Doors are double doors of coord [[x1,x2], [y1,y2]]
		'DOORS': [
			[[5, 6, 7], [19, 19, 19]],
			[[2, 3, 4], [15, 15, 15]],
			[[5, 6, 7], [11, 11, 11]],
			[[2, 3, 4], [7, 7, 7]],
			[[5, 6, 7], [3, 3, 3]]
		],

		'PLATES': [
			[7, 21],
			[2, 17],
			[7, 13],
			[2, 9],
			[7, 5]
		],

		'AGENTS': [
			[6, 21],
			[6, 20],
			[5, 21],
			[5, 20],
			[4, 21],
			[4, 20]
		],

		'GOAL': [
			[3, 1]
		]
	}
}
