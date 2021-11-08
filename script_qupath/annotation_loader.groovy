import com.google.gson.GsonBuilder
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.common.ColorTools
import qupath.lib.gui.dialogs.Dialogs


// Remove all detections
removeObjects(getDetectionObjects().findAll(), true)

// Dialog to load JSON detection file
path = Dialogs.getChooser(null).promptForFile(
    "Load detection JSON file", null, "JSON files", new String[]{".json"})

// Create reader and read file content
println("Reading ...")
Reader reader = new FileReader(path)
data = reader.readLines()[0]

// Parse file content
println("Parsing ...")
def gson = new GsonBuilder()
        .setPrettyPrinting()
        .create()
data = gson.fromJson(data, Map.class);

// Crate object plane
def plane = ImagePlane.getPlane(0, 0)

// Iterate over detections
println("Number of detections:" + data.size())
println("Create objects ...")
objs = []
data.each{ key, value ->
    // Define color
    color = ColorTools.packRGB(value["color"][0].intValue(), value["color"][1].intValue(), value["color"][2].intValue())
    // Define class
    cls = PathClassFactory.getPathClass(value["class"], color)
    roi = ROIs.createPolygonROI(value["coords"][0] as double[], value["coords"][1] as double[], plane)
    // obj = PathObjects.createAnnotationObject(roi, cls)
    objs << PathObjects.createDetectionObject(roi, cls)
}

println("Append objects ...")
addObjects(objs)
println("Done!")