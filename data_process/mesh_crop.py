import numpy as np
import vtk
import os


def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == '.ply':
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.vtp':
        reader = vtk.vtkXMLpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.obj':
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.stl':
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.vtk':
        reader = vtk.vtkpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == '.g':
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        poly_data = None
    return poly_data

def main(processed_dir):
    
    
    colors = vtk.vtkNamedColors()
    backgroundColor = colors.GetColor3d('steel_blue')
    
    filePath = os.path.join(processed_dir,  'fusion_mesh.ply')
    outPut = os.path.join(processed_dir, 'fusion_mesh_foreground.ply')
    
    if filePath and os.path.isfile(filePath):
        polyData = ReadPolyData(filePath)
    else:
        print("there is not exist polyData")
        return 0
        
    # plane = vtk.vtkPlane()
    # plane.SetOrigin(polyData.GetCenter())
    # plane.SetNormal(1.0, 0, 0)
    
    # crop box 
    box  = vtk.vtkBox()
    box.SetBounds(-0.4,0.4,-0.4,0.4, 0.015 ,0.2)
    
    #  clipper  
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polyData)
    clipper.SetClipFunction(box)
    clipper.SetInsideOut(-1)
    #clipper.SetGenerateClippedOutput(False)
    clipper.SetValue(0)
    clipper.Update()
    
    # get the cropped data
    polyDataCropped = clipper.GetOutput()
    #print(polyDataCropped)
    left = clipper. GenerateClippedOutputOn()
    
    # write to file
    plyWriter = vtk.vtkPLYWriter()
    plyWriter.SetFileName(outPut)
    #plyWriter.setInput(polyDataCropped)
    #plyWriter.SetInputConnection(polyDataCropped.GetOutput())
    plyWriter.SetInputData(polyDataCropped)
    plyWriter.Write()
    
    ##      to show 
    #   mapper  
    clipMapper = vtk.vtkDataSetMapper()
    clipMapper.SetInputData(polyDataCropped)
    ##  Actor
    clipActor = vtk.vtkActor()
    clipActor.SetMapper(clipMapper)
    
    ## renderer and render window
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    
    renderer.SetBackground(backgroundColor)
    renderWindow.SetSize(1024,768)
    
    renderer.AddActor(clipActor)
    
    # Generate an interesting view
    renderer.ResetCamera()
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(30)
    renderer.GetActiveCamera().Dolly(1.2)
    renderer.ResetCameraClippingRange()

    renderWindow.Render()
    renderWindow.SetWindowName('CapClip')
    renderWindow.Render()

    interactor.Start()



if __name__ == '__main__':
    
    processed_dir = '/home/cyn/dataset/dense-net-entire/pdc/logs_proto/000111_1/processed'
    main(processed_dir)