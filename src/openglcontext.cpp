#include "openglcontext.hpp"
#include <sstream>
#include <map>
using namespace std;

// Create an OpenGL context
OpenGLContext::OpenGLContext() {
	const int major = 4, minor = 6;

	// Configuration attributes
	static const EGLint configAttribs[] = {
		EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_NONE
	};

	// Context attributes
	const EGLint contextAttribs[] = {
		EGL_CONTEXT_MAJOR_VERSION, major,
		EGL_CONTEXT_MINOR_VERSION, minor,
		EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
		EGL_NONE
	};

	// Get default display
	eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	checkEGLError("eglGetDisplay");
	if (eglDpy == EGL_NO_DISPLAY)
		throw runtime_error("Failed to get default EGL display");

	// Initialize EGL
	EGLint eglMajor, eglMinor;
	EGLBoolean ret = eglInitialize(eglDpy, &eglMajor, &eglMinor);
	checkEGLError("eglInitialize");
	if (ret != EGL_TRUE)
		throw runtime_error("Failed to initialize EGL");

	// Select an appropriate configuration
	EGLint numConfigs;
	EGLConfig eglCfg;
	eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
	checkEGLError("eglChooseConfig");

	// Bind the API
	eglBindAPI(EGL_OPENGL_API);
	checkEGLError("eglBindAPI");

	// Create the context
	eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, contextAttribs);
	checkEGLError("eglCreateContext");
	if (eglCtx == EGL_NO_CONTEXT)
		throw runtime_error("Failed to create context");

	// Make the context current
	makeCurrent();
}

OpenGLContext::~OpenGLContext() {
	// Release any held resources
	for (auto v : vaos)
		glDeleteVertexArrays(1, &v);
	vaos.clear();

	for (auto b : buffers)
		glDeleteBuffers(1, &b);
	buffers.clear();

	for (auto t : textures)
		glDeleteTextures(1, &t);
	textures.clear();

	for (auto f : framebuffers)
		glDeleteFramebuffers(1, &f);
	framebuffers.clear();

	for (auto r : renderbuffers)
		glDeleteRenderbuffers(1, &r);
	renderbuffers.clear();

	for (auto s : shaders)
		glDeleteShader(s);
	shaders.clear();

	for (auto p : programs)
		glDeleteProgram(p);
	programs.clear();

	// Make not current anymore
	eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

	// Terminate display connection
	eglTerminate(eglDpy);
}

void OpenGLContext::makeCurrent() {
	// Make this context current
	eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);
	checkEGLError("eglMakeCurrent");
}

bool OpenGLContext::isCurrent() {
	// Get current context
	EGLContext ctx = eglGetCurrentContext();
	checkEGLError("eglGetCurrentContext");
	return (ctx == eglCtx);
}

// Create vertex array object
GLuint OpenGLContext::genVAO() {
	GLuint vao;
	glGenVertexArrays(1, &vao);
	vaos.insert(vao);
	return vao;
}
// Create buffer object
GLuint OpenGLContext::genBuffer() {
	GLuint buf;
	glGenBuffers(1, &buf);
	buffers.insert(buf);
	return buf;
}
// Create texture object
GLuint OpenGLContext::genTexture() {
	GLuint tex;
	glGenTextures(1, &tex);
	textures.insert(tex);
	return tex;
}
// Create framebuffer object
GLuint OpenGLContext::genFramebuffer() {
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	framebuffers.insert(fbo);
	return fbo;
}
// Create renderbuffer object
GLuint OpenGLContext::genRenderbuffer() {
	GLuint rbo;
	glGenRenderbuffers(1, &rbo);
	renderbuffers.insert(rbo);
	return rbo;
}
// Compile a shader object from source code
GLuint OpenGLContext::compileShader(GLenum type, string source) {
	const char* source_cstr = source.c_str();
	GLint length = source.length();

	// Compile the shader
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source_cstr, &length);
	glCompileShader(shader);

	// Make sure compilation succeeded
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		// Compilation failed, get the info log
		GLint logLength;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
		vector<GLchar> logText(logLength);
		glGetShaderInfoLog(shader, logLength, NULL, logText.data());

		// Construct an error message with the compile log
		stringstream ss;
		string typeStr = "";
		switch (type) {
		case GL_VERTEX_SHADER:
			typeStr = "vertex"; break;
		case GL_GEOMETRY_SHADER:
			typeStr = "geometry"; break;
		case GL_FRAGMENT_SHADER:
			typeStr = "fragment"; break;
		}
		ss << "Error compiling " << typeStr << " shader!" << endl << endl;
		ss << logText.data() << endl;

		// Cleanup shader and throw an exception
		glDeleteShader(shader);
		throw runtime_error(ss.str());
	}

	// Add to internal storage and return
	shaders.insert(shader);
	return shader;
}
// Link shader objects together to create a program
GLuint OpenGLContext::linkProgram(vector<GLuint> shaderObjs) {
	GLuint program = glCreateProgram();

	// Attach the shaders and link the program
	for (auto s : shaderObjs)
		glAttachShader(program, s);
	glLinkProgram(program);

	// Detach shaders
	for (auto s : shaderObjs)
		glDetachShader(program, s);

	// Make sure link succeeded
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		// Link failed, get the info log
		GLint logLength;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
		vector<GLchar> logText(logLength);
		glGetProgramInfoLog(program, logLength, NULL, logText.data());

		// Construct an error message with the compile log
		stringstream ss;
		ss << "Error linking program!" << endl << endl;
		ss << logText.data() << endl;

		// Cleanup program and throw an exception
		glDeleteProgram(program);
		throw runtime_error(ss.str());
	}

	// Add to internal storage and return
	programs.insert(program);
	return program;
}

// Delete vertex array object
void OpenGLContext::deleteVAO(GLuint vao) {
	if (vaos.erase(vao))
		glDeleteVertexArrays(1, &vao);
}
// Delete buffer object
void OpenGLContext::deleteBuffer(GLuint buffer) {
	if (buffers.erase(buffer))
		glDeleteBuffers(1, &buffer);
}
// Delete texture object
void OpenGLContext::deleteTexture(GLuint texture) {
	if (textures.erase(texture))
		glDeleteTextures(1, &texture);
}
// Delete framebuffer object
void OpenGLContext::deleteFramebuffer(GLuint framebuffer) {
	if (framebuffers.erase(framebuffer))
		glDeleteFramebuffers(1, &framebuffer);
}
// Delete renderbuffer object
void OpenGLContext::deleteRenderbuffer(GLuint renderbuffer) {
	if (renderbuffers.erase(renderbuffer))
		glDeleteRenderbuffers(1, &renderbuffer);
}
// Delete shader object
void OpenGLContext::deleteShader(GLuint shader) {
	if (shaders.erase(shader))
		glDeleteShader(shader);
}
// Delete program object
void OpenGLContext::deleteProgram(GLuint program) {
	if (programs.erase(program))
		glDeleteProgram(program);
}

// Reads the EGL error state and throws an exception if not EGL_SUCCESS
void OpenGLContext::checkEGLError(string what) {
	string prepend;
	if (!what.empty())
		prepend = what + ": ";

	// Map error constants to strings
	static const map<EGLint, string> errStrMap = {
		{ EGL_NOT_INITIALIZED, "EGL_NOT_INITIALIZED" },
		{ EGL_BAD_ACCESS, "EGL_BAD_ACCESS" },
		{ EGL_BAD_ALLOC, "EGL_BAD_ALLOC" },
		{ EGL_BAD_ATTRIBUTE, "EGL_BAD_ATTRIBUTE" },
		{ EGL_BAD_CONTEXT, "EGL_BAD_CONTEXT" },
		{ EGL_BAD_CONFIG, "EGL_BAD_CONFIG" },
		{ EGL_BAD_CURRENT_SURFACE, "EGL_BAD_CURRENT_SURFACE" },
		{ EGL_BAD_DISPLAY, "EGL_BAD_DISPLAY" },
		{ EGL_BAD_SURFACE, "EGL_BAD_SURFACE" },
		{ EGL_BAD_MATCH, "EGL_BAD_MATCH" },
		{ EGL_BAD_PARAMETER, "EGL_BAD_PARAMETER" },
		{ EGL_BAD_NATIVE_PIXMAP, "EGL_BAD_NATIVE_PIXMAP" },
		{ EGL_BAD_NATIVE_WINDOW, "EGL_BAD_NATIVE_WINDOW" },
		{ EGL_CONTEXT_LOST, "EGL_CONTEXT_LOST" },
	};

	// Get the error, return if no error
	EGLint err = eglGetError();
	if (err == EGL_SUCCESS) return;

	// Find the error code in the string map
	auto errIt = errStrMap.find(err);
	if (errIt != errStrMap.end())
		throw runtime_error(prepend + errIt->second);
	else
		// Error code not found,
		throw runtime_error(prepend + "unknown error: " + to_string(err));
}
