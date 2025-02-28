import xml.etree.ElementTree as ET
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet
import utils as u


class Race:
    """Class to parse and retrieve course data"""
    def __init__(self, creation_time, race_start_time, race_id, race_type,
                 crew_number, yacht_settings, participants,
                 compound_marks, compound_mark_sequence, course_limits):
        self.creation_time = creation_time
        self.race_start_time = pd.Timestamp(race_start_time)
        self.race_id = race_id
        self.race_type = race_type
        self.crew_number = crew_number
        self.yacht_settings = yacht_settings
        self.participants = participants
        self.compound_marks = compound_marks
        self.compound_mark_sequence = compound_mark_sequence
        self.course_limits = course_limits

    @classmethod
    def from_xml(cls, xml_file):
        """Parses the given XML file and returns a Race instance"""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Basic race metadata
        creation_time = root.find('CreationTimeDate').text
        race_start_time = root.find('RaceStartTime').attrib.get('Start')
        race_id = root.find('RaceID').text
        race_type = root.find('RaceType').text

        # Settings: Crew and Yacht
        settings = root.find('Settings')
        crew_elem = settings.find('Crew')
        crew_number = crew_elem.attrib.get('Number') if crew_elem is not None else None
        yacht_elem = settings.find('Yacht')
        yacht_settings = yacht_elem.attrib if yacht_elem is not None else {}

        # Participants
        participants = []
        participants_elem = root.find('Participants')
        if participants_elem is not None:
            for yacht in participants_elem.findall('Yacht'):
                participants.append(yacht.attrib.get('SourceID'))

        # Course: Compound Marks and their Marks
        compound_marks = {}
        course_elem = root.find('Course')
        wg_marks = []
        lg_marks = []
        if course_elem is not None:
            for compound in course_elem.findall('CompoundMark'):
                cm_id = compound.attrib.get('CompoundMarkID')
                marks = []
                for mark in compound.findall('Mark'):
                    if mark.attrib.get('Name') == 'WG1' or mark.attrib.get('Name') == 'WG2':
                        wg_marks.append({'Lat': mark.attrib.get('TargetLat'), 'Lon': mark.attrib.get('TargetLng')})
                    elif mark.attrib.get('Name') == 'LG1' or mark.attrib.get('Name') == 'LG2':
                        lg_marks.append({'Lat': mark.attrib.get('TargetLat'), 'Lon': mark.attrib.get('TargetLng')})
                    marks.append({
                        'SeqID': mark.attrib.get('SeqID'),
                        'Name': mark.attrib.get('Name'),
                        'TargetLat': float(mark.attrib.get('TargetLat')),
                        'TargetLng': float(mark.attrib.get('TargetLng')),
                        'SourceID': mark.attrib.get('SourceID'),
                    })
                compound_marks[cm_id] = {
                    'Name': compound.attrib.get('Name'),
                    'Marks': marks
                }

        # Calculate mean position of WG and LG marks
        upwind_midpoint = calculate_midpoint(wg_marks)
        bottom_midpoint = calculate_midpoint(lg_marks)
        if upwind_midpoint is not None and bottom_midpoint is not None:
            axis_bearing = u.calculate_bearing(bottom_midpoint[0], bottom_midpoint[1],
                                               upwind_midpoint[0], upwind_midpoint[1])
            compound_marks['CourseAxis'] = {
                'UpwindMidpoint': upwind_midpoint,
                'BottomMidpoint': bottom_midpoint,
                'Bearing': axis_bearing
            }

        # Compound Mark Sequence
        compound_mark_sequence = []
        seq_elem = root.find('CompoundMarkSequence')
        if seq_elem is not None:
            for corner in seq_elem.findall('Corner'):
                compound_mark_sequence.append({
                    'SeqID': corner.attrib.get('SeqID'),
                    'CompoundMarkID': corner.attrib.get('CompoundMarkID'),
                    'Rounding': corner.attrib.get('Rounding'),
                    'ZoneSize': corner.attrib.get('ZoneSize'),
                })

        # Boundaries
        course_limits = {}
        for course_limit in root.findall('CourseLimit'):
            name = course_limit.attrib.get('name')
            limits = []
            for limit in course_limit.findall('Limit'):
                limits.append({
                    'SeqID': limit.attrib.get('SeqID'),
                    'Lat': float(limit.attrib.get('Lat')),
                    'Lon': float(limit.attrib.get('Lon')),
                })
            course_limits[name] = {
                'draw': course_limit.attrib.get('draw'),
                'avoid': course_limit.attrib.get('avoid'),
                'fill': course_limit.attrib.get('fill'),
                'lock': course_limit.attrib.get('lock'),
                'colour': course_limit.attrib.get('colour'),
                'notes': course_limit.attrib.get('notes'),
                'Limits': limits
            }

        return cls(creation_time, race_start_time, race_id, race_type,
                   crew_number, yacht_settings, participants,
                   compound_marks, compound_mark_sequence, course_limits)

    def plot_course(self):
        """
        Plots the course of the race using Bokeh.
        """
        # Create a new figure
        p = figure(title="Race Course", x_axis_label="Longitude", y_axis_label="Latitude",
                   width=800, height=600)

        mark_x = []
        mark_y = []
        mark_ids = []
        for compound in self.compound_marks.values():
            if 'Marks' not in compound:
                continue
            for mark in compound['Marks']:
                try:
                    x = float(mark.get('TargetLng', 0))
                    y = float(mark.get('TargetLat', 0))
                except (TypeError, ValueError):
                    continue
                mark_x.append(x)
                mark_y.append(y)
                mark_ids.append(mark.get("Name"))

        p.scatter(x=mark_x, y=mark_y, size=10, color="navy", alpha=0.5)

        source = ColumnDataSource(data=dict(x=mark_x, y=mark_y, id=mark_ids))
        labels = LabelSet(x='x', y='y', text='id', level='glyph',
                          x_offset=5, y_offset=5, source=source)
        p.add_layout(labels)

        if "Boundary" in self.course_limits:
            boundary = self.course_limits["Boundary"]
            poly_x = []
            poly_y = []
            for limit in boundary.get("Limits", []):
                try:
                    poly_x.append(float(limit.get("Lon", 0)))
                    poly_y.append(float(limit.get("Lat", 0)))
                except (TypeError, ValueError):
                    continue
            # Ensure the polygon is closed
            if poly_x and poly_y and (poly_x[0] != poly_x[-1] or poly_y[0] != poly_y[-1]):
                poly_x.append(poly_x[0])
                poly_y.append(poly_y[0])
            p.patch(x=poly_x, y=poly_y, color="green", fill_alpha=0.2, line_width=2, legend_label="Boundary")

        output_file("race_course.html")
        show(p)


    def __str__(self):
        return f"Race {self.race_id} starting at {self.race_start_time}"


def calculate_midpoint(points):
    """
    Given a list of points (each a dict with 'Lat' and 'Lon'), return the average point.
    """
    if not points:
        return None
    avg_lat = sum(float(p['Lat']) for p in points) / len(points)
    avg_lon = sum(float(p['Lon']) for p in points) / len(points)
    return avg_lat, avg_lon