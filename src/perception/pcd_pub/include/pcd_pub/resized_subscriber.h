#include <image_transport/simple_subscriber_plugin.h>
#include <pcd_pub/ResizedImage.h>

class ResizedSubscriber : public image_transport::SimpleSubscriberPlugin<pcd_pub::ResizedImage>
{
public:
  virtual ~ResizedSubscriber() {}

  virtual std::string getTransportName() const
  {
    return "resized";
  }

protected:
  virtual void internalCallback(const typename pcd_pub::ResizedImage::ConstPtr& message,
                                const Callback& user_cb);
};
