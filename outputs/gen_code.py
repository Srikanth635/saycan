```python
# Define the target locations for picking up the apple and placing it on the kitchen island surface
apple_pickup_location = Pose(x=1.0, y=0.5, z=0.0, orientation=Quaternion(0.0, 0.0, 0.0, 1.0))
kitchen_island_surface_location = Pose(x=2.0, y=1.0, z=0.0, orientation=Quaternion(0.0, 0.0, 0.0, 1.0))

# Create the action designators for picking up the apple and placing it on the kitchen island surface
pickup_apple_action = PickUpAction(object_designator_description=ObjectDesignatorDescription(names=["apple"]),
                                    arms=["left", "right"], grasps=["top"])
place_on_kitchen_island_action = PlaceAction(object_designator_description=ObjectDesignatorDescription(names=["apple"]),
                                             target_locations=[kitchen_island_surface_location], arms=["left", "right"])

# Ground the action designators to get the performable actions
pickup_apple_performable = pickup_apple_action.ground()
place_on_kitchen_island_performable = place_on_kitchen_island_action.ground()

# Plan the navigation for the robot if needed
navigation_to_apple = NavigateAction(target_locations=[apple_pickup_location])
navigation_to_kitchen_island = NavigateAction(target_locations=[kitchen_island_surface_location])

# Ground the navigation actions to get the performable actions
navigation_to_apple_performable = navigation_to_apple.ground()
navigation_to_kitchen_island_performable = navigation_to_kitchen_island.ground()

# Perform the actions in sequence
navigation_to_apple_performable.perform()
pickup_apple_performable.perform()
navigation_to_kitchen_island_performable.perform()
place_on_kitchen_island_performable.perform()
```