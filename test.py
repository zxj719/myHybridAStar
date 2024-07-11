class Map:

    xyResolution: int = 4     # grid block length
    yawResolution: float =15.0 # grid block possible yaws

    def __init__(self) -> None:
        # Build Map
        self.obstacleX, self.obstacleY = [], []
        for i in range(51):
            self.obstacleX.append(i)
            self.obstacleY.append(0)

        for i in range(51):
            self.obstacleX.append(0)
            self.obstacleY.append(i)

        for i in range(51):
            self.obstacleX.append(i)
            self.obstacleY.append(50)

        for i in range(51):
            self.obstacleX.append(50)
            self.obstacleY.append(i)
        
        for i in range(6,20):
            self.obstacleX.append(i)
            self.obstacleY.append(30)

        for i in range(30,51):
            self.obstacleX.append(i)
            self.obstacleY.append(30) 

        for i in range(0,31):
            self.obstacleX.append(20)
            self.obstacleY.append(i) 

        for i in range(0,31):
            self.obstacleX.append(30)
            self.obstacleY.append(i) 

        for i in range(40,50):
            self.obstacleX.append(15)
            self.obstacleY.append(i)

        for i in range(25,40):
            self.obstacleX.append(i)
            self.obstacleY.append(35)

        # calculate min max map grid index based on obstacles in map
        self.mapMinX, self.mapMinY = self.grid_idx(min(self.obstacleX), min(self.obstacleY))
        self.mapMaxX = round(max(self.obstacleX) / self.xyResolution)
        self.mapMaxY = round(max(self.obstacleY) / self.xyResolution)

        # create a KDTree to represent obstacles
        # self.ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(self.obstacleX, self.obstacleY)])

    @staticmethod
    def grid_idx(*pos):
        """Calculate (x,y) --> map grid index (x_idx, y_idx)"""
        return (round(pos[0] / Map.xyResolution), round(pos[1] / Map.xyResolution))

mapParameters = Map()
print(mapParameters.mapMinY)