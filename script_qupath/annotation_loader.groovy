import com.google.gson.GsonBuilder
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.common.ColorTools

// Define output path
def folder_json = "/media/abbet/Data/pub/current/cls/stroma_roi_1000_supervised"
def suffix_json = "_classification_supervised_r50_kather19_stroma_roi_1000_supervised_detection.json"

// Load slide name
def filename = getCurrentImageData().getServer().getURIs().getAt(0).getPath()
filename = filename.substring(filename.lastIndexOf("/")+1)
path = folder_json +  "/" + filename + suffix_json

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

// Define class
cls = PathClassFactory.getPathClass("StromaROI", ColorTools.packRGB(51, 153, 51))

// Iterate over detections
println("Number of detections:" + data.size())
println("Create objects ...")
objs = []
data.each{ key, value ->
    color = ColorTools.packRGB(value["color"][0].intValue(), value["color"][1].intValue(), value["color"][2].intValue())
    cls = PathClassFactory.getPathClass(value["class"], color)
    roi = ROIs.createPolygonROI(value["coords"][0] as double[], value["coords"][1] as double[], plane)
    // obj = PathObjects.createAnnotationObject(roi, cls)
    objs << PathObjects.createDetectionObject(roi, cls)
}

println("Append objects ...")
addObjects(objs)
println("Done!")