plan = """
self.go_to("drawer")
self.open_drawer()
detections = self.get_detections()
for detection in detections:
    self.pick(detection)
    self.place("drawer")
self.close_drawer()

self.move_object_closest_to("table", "sofa")
success = self.find_and_align_to_object("teddy_bear")
if success:
    self.pick("teddy_bear")
    self.move_between_objects("potted plant", "sofa")
    self.go_to("user")
    self.place("table")
    self.speak("I'm done.")
else:
    self.speak("I can't find the teddy bear.")
"""

# Split the plan into lines
lines = plan.split('\n')
print(lines)

# Execute each line with correct indentation
for line in lines:
    # exec(line.strip()+ '\n')
    print(line)